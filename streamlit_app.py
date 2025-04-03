"""Main application file for Brandfolder Creative Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
from anthropic import Anthropic
from fuzzywuzzy import fuzz
from PIL import Image
import zipfile
import io
import tempfile
import subprocess
from io import BytesIO

# Set up your API key
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env
api_key = os.getenv('api_key')
client = Anthropic(api_key=api_key)
import requests

# Streamlit app title
st.title("Brandfolder Creative Analysis")

# File uploaders with instructions
st.sidebar.header("Upload Files")

st.sidebar.markdown("**Brandfolder Zip**")
brandfolder_zip = st.sidebar.file_uploader("Upload Brandfolder Zip", type=["zip"])
st.sidebar.markdown("Select all files you'd like to include in Brandfolder and download as zip")

st.sidebar.markdown("**Brandfolder CSV**")
brandfolder_csv = st.sidebar.file_uploader("Upload Brandfolder CSV", type=["csv"])
st.sidebar.markdown("Select all files you'd like to include in Brandfolder and download as a csv")

st.sidebar.markdown("**Performance Data XLSX**")
performance_data = st.sidebar.file_uploader("Upload Performance Data XLSX", type=["xlsx"])
st.sidebar.markdown("Utilize Pivot Tables in MCI with the following variables: Platforms, Campaign Name, Media Buy (i.e. Audience), Creative Name, Media Cost, Impressions, Clicks, and any other KPI you'd like to use.")
st.sidebar.markdown("Creative Name must have the brandfolder key in the naming convention.")

if brandfolder_zip and brandfolder_csv and performance_data:
    df_brandfolder = pd.read_csv(brandfolder_csv)
    df_performance = pd.read_excel(performance_data)
    
    # Function to convert currency strings to float
    def convert_currency_to_float(x):
        if isinstance(x, str):
            return float(x.replace('$', '').replace(',', ''))
        else:
            return x

    # Apply the conversion function to numeric columns only
    numeric_columns = ['Media Cost', 'Impressions', 'Clicks']  # Add other numeric columns here

    for col in numeric_columns:
        if col in df_performance.columns:
            try:
                df_performance[col] = df_performance[col].apply(convert_currency_to_float)
            except ValueError:
                print(f"Skipping column '{col}' as it cannot be converted to float.")
    
    # Extract Brandfolder Key from Creative Name
    df_performance['Brandfolder Key'] = df_performance['Creative Name'].apply(lambda x: x.split('_')[-1] if pd.notnull(x) else None)
    
    # Left join the dataframes
    merged_df = pd.merge(df_performance, df_brandfolder, left_on='Brandfolder Key', right_on='key', how='inner')
    
    st.write("### Merged Data Preview")
    st.write(merged_df.head())
    
    # Provide download link for merged file as XLSX
    @st.cache_data
    def convert_df_to_excel(df):
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Merged Data')
        processed_data = output.getvalue()
        return processed_data

    excel_data = convert_df_to_excel(merged_df)
    st.download_button("Download Merged XLSX", excel_data, "merged_file.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Extract numeric columns from df_performance
    numeric_columns = df_performance.select_dtypes(include=['number']).columns.tolist()

    # Create a dropdown for selecting a numeric KPI
    st.write("### Select a KPI to Analyze")
    selected_kpi = st.selectbox("Select a KPI", numeric_columns)

    # Create a dropdown for selecting grouping method
    st.write("### Select Performance Grouping")
    grouping_options = [
        "Aggregate (All Data)", 
        "Group by Platforms", 
        "Group by Media Buy Name", 
        "Group by Platforms and Media Buy Name"
    ]
    selected_grouping = st.selectbox("Select how to group the performance data", grouping_options)

    # Function to convert .mov files to .mp4
    def convert_mov_to_mp4(zip_file):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip contents
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Walk through the extracted files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith('.mov'):
                        mov_path = os.path.join(root, file)
                        mp4_path = mov_path.rsplit('.', 1)[0] + '.mp4'

                        try:
                            # Use ffmpeg to convert .mov to .mp4
                            subprocess.run([
                                'ffmpeg',
                                '-i', mov_path,
                                '-vcodec', 'libx264',
                                '-acodec', 'aac',
                                mp4_path
                            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                            os.remove(mov_path)  # Remove original .mov
                        except subprocess.CalledProcessError as e:
                            st.error(f"Error converting {file} with ffmpeg.")

            # Re-zip contents into a new zip in memory
            mem_zip = BytesIO()
            with zipfile.ZipFile(mem_zip, 'w', zipfile.ZIP_DEFLATED) as new_zip:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, temp_dir)
                        new_zip.write(full_path, rel_path)

            mem_zip.seek(0)
            return mem_zip
    
    brandfolder_zip = convert_mov_to_mp4(brandfolder_zip)
    
    # Function to properly calculate CP metrics based on their components
    def calculate_cp_metric(df, metric_name):
        # Define the components needed for each CP metric
        cp_components = {
            'CPC': {'numerator': 'Media Cost', 'denominator': 'Clicks', 'multiplier': 1},
            'CPM': {'numerator': 'Media Cost', 'denominator': 'Impressions', 'multiplier': 1000},
            # Add other CP metrics as needed
        }
        
        # Default for any CP metric not explicitly defined
        default_components = {'numerator': 'Media Cost', 'denominator': 'Impressions', 'multiplier': 1}
        
        # Get the components for this metric
        components = cp_components.get(metric_name, default_components)
        
        # Calculate the CP metric
        numerator_sum = df[components['numerator']].sum()
        denominator_sum = df[components['denominator']].sum()
        
        if denominator_sum > 0:
            return (numerator_sum / denominator_sum) * components['multiplier']
        else:
            return np.nan

    # Function to match creative name to image
    def find_closest_matching_creative(creative_name, uploaded_zip):
        valid_image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
        valid_video_extensions = ('.mp4', '.mov', '.webm')

        try:
            with zipfile.ZipFile(uploaded_zip) as outer_zip:
                file_list = outer_zip.namelist()

                # 1. Try to match images
                image_files = [f for f in file_list if f.lower().endswith(valid_image_extensions)]
                if image_files:
                    best_image_match = max(image_files, key=lambda f: fuzz.ratio(creative_name.lower(), os.path.basename(f).lower()))
                    best_image_score = fuzz.ratio(creative_name.lower(), os.path.basename(best_image_match).lower())

                    if best_image_score > 60:
                        with outer_zip.open(best_image_match) as img_file:
                            try:
                                return 'image', Image.open(io.BytesIO(img_file.read()))
                            except Exception as e:
                                print(f"Error loading image '{best_image_match}': {e}")

                # 2. Try to match video files
                video_files = [f for f in file_list if f.lower().endswith(valid_video_extensions)]
                if video_files:
                    best_video_match = max(video_files, key=lambda f: fuzz.ratio(creative_name.lower(), os.path.basename(f).lower()))
                    best_video_score = fuzz.ratio(creative_name.lower(), os.path.basename(best_video_match).lower())

                    if best_video_score > 60:
                        with outer_zip.open(best_video_match) as video_file:
                            video_bytes = video_file.read()
                            return 'video', video_bytes

                # 3. Try to match nested ZIPs and extract index.html
                nested_zips = [f for f in file_list if f.lower().endswith(".zip")]
                for nested_zip_path in nested_zips:
                    zip_base_name = os.path.basename(nested_zip_path).rsplit(".", 1)[0]
                    match_score = fuzz.ratio(creative_name.lower(), zip_base_name.lower())
                    print(f"Trying nested zip: {nested_zip_path}, match score: {match_score}")

                    if match_score > 60:
                        try:
                            with outer_zip.open(nested_zip_path) as nested_zip_bytes:
                                with zipfile.ZipFile(io.BytesIO(nested_zip_bytes.read())) as nested_zip:
                                    temp_dir = tempfile.mkdtemp(prefix="creative_")
                                    nested_zip.extractall(temp_dir)

                                    print(f"Extracted nested zip to: {temp_dir}")
                                    
                                    # Walk the extracted contents
                                    for root, dirs, files in os.walk(temp_dir):
                                        for file in files:
                                            print(f"Found file in nested zip: {file}")
                                            if file.lower() == "index.html":
                                                index_path = os.path.join(root, file)
                                                print(f"Found index.html: {index_path}")
                                                return 'html_link', index_path
                        except Exception as e:
                            print(f"Failed to read nested zip '{nested_zip_path}': {e}")

                return None, None

        except Exception as e:
            print(f"Zip handling error: {e}")
            return None, None
    
    # Function to display performance for a group
    def display_performance(df, group_name=""):
        # Check if df is empty or the selected KPI has all NaN values
        if df.empty or df[selected_kpi].isna().all():
            st.write(f"**{group_name}**")
            st.write("No valid data available for this group.")
            st.write("---")
            return
        
        # Remove rows with NaN values in the selected KPI
        df_valid = df.dropna(subset=[selected_kpi])
        
        if df_valid.empty:
            st.write(f"**{group_name}**")
            st.write("No valid data available for this group.")
            st.write("---")
            return
        
        # Group by 'name' to get creative-level data
        grouped_by_creative = df_valid.groupby('name')
        
        # Create a DataFrame to hold recalculated metrics for each creative
        creative_metrics = []
        
        for creative_name, creative_data in grouped_by_creative:
            if selected_kpi.startswith('CP'):
                # Recalculate the CP metric correctly
                metric_value = calculate_cp_metric(creative_data, selected_kpi)
                creative_metrics.append({
                    'name': creative_name,
                    selected_kpi: metric_value
                })
            else:
                # For non-CP metrics, just sum the values
                creative_metrics.append({
                    'name': creative_name,
                    selected_kpi: creative_data[selected_kpi].sum()
                })
        
        # Convert to DataFrame
        creative_metrics_df = pd.DataFrame(creative_metrics)
        
        # Sort based on the metric type
        if selected_kpi.startswith('CP'):
            # For CP metrics, lower is better
            sorted_df = creative_metrics_df.sort_values(by=selected_kpi, na_position='last')
            best_performers = sorted_df.head(3)  # lowest values
            worst_performers = sorted_df.tail(3)  # highest values
        else:
            # For other metrics, higher is better
            sorted_df = creative_metrics_df.sort_values(by=selected_kpi, ascending=False, na_position='last')
            best_performers = sorted_df.head(3)  # highest values
            worst_performers = sorted_df.tail(3)  # lowest values
        
        st.write(f"**{group_name}**")
        
        st.write("#### Best Performing Creatives:")
        for index, row in best_performers.iterrows():
            metric_value = row[selected_kpi]
            if pd.notnull(metric_value):
                formatted_value = f"{metric_value:.2f}" if isinstance(metric_value, (float, int)) else metric_value
                st.write(f"- **Creative Name:** {row['name']}, **{selected_kpi}:** {formatted_value}")
            match_type, content = find_closest_matching_creative(row["name"], brandfolder_zip)

            if match_type == "image":
                st.image(content, caption=row["name"], use_column_width=True)

            elif match_type == "video":
                st.video(content)

            elif match_type == "html_link":
                html_file_path = os.path.abspath(content)
                st.markdown(
                    f'[🔗 Copy and Paste link to open Animated Creative in New Tab](file://{html_file_path})',
                    unsafe_allow_html=True
                )

            else:
                st.warning("No matching creative found.")
        
        st.write("#### Worst Performing Creatives:")
        for index, row in worst_performers.iterrows():
            metric_value = row[selected_kpi]
            if pd.notnull(metric_value):
                formatted_value = f"{metric_value:.2f}" if isinstance(metric_value, (float, int)) else metric_value
                st.write(f"- **Creative Name:** {row['name']}, **{selected_kpi}:** {formatted_value}")
                match_type, content = find_closest_matching_creative(row["name"], brandfolder_zip)

                if match_type == "image":
                    st.image(content, caption=row["name"], use_column_width=True)

                elif match_type == "video":
                    st.video(content)

                elif match_type == "html_link":
                    html_file_path = os.path.abspath(content)
                    st.markdown(
                        f'[🔗 Copy and Paste link to open Animated Creative in New Tab](file://{html_file_path})',
                        unsafe_allow_html=True
                    )

                else:
                    st.warning("No matching creative found.")
                        
        st.write("---")

    # Function to generate insights using Claude
    def generate_insights(data, selected_kpi):
        focus_variables = ["Tags", "Asset Type", "Creative Content"]
        prompt = f"Analyze the following data focusing on {', '.join(focus_variables)} and the KPI {selected_kpi}. Determine which characteristics of the creatives work best. If the KPI starts with 'CP', best means the lowest value; otherwise, it means the highest value. The data is: {data.to_dict(orient='records')}"
        
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

    # Display performance based on selected grouping
    st.write("### Best and Worst Performing Creatives")
    
    if selected_grouping == "Aggregate (All Data)":
        display_performance(merged_df, "All Data")
        insights = generate_insights(merged_df, selected_kpi)
        st.write("### AI Insights")
        st.write(insights)
        
    elif selected_grouping == "Group by Platforms":
        for platform in merged_df['Platforms'].dropna().unique():
            platform_df = merged_df[merged_df['Platforms'] == platform]
            display_performance(platform_df, f"Platforms: {platform}")
            insights = generate_insights(platform_df, selected_kpi)
            st.write(f"### AI Insights for {platform}")
            st.write(insights)
            
    elif selected_grouping == "Group by Media Buy Name":
        for media_buy_name in merged_df['Media Buy Name'].dropna().unique():
            media_buy_df = merged_df[merged_df['Media Buy Name'] == media_buy_name]
            display_performance(media_buy_df, f"Media Buy Name: {media_buy_name}")
            insights = generate_insights(media_buy_df, selected_kpi)
            st.write(f"### AI Insights for {media_buy_name}")
            st.write(insights)
            
    elif selected_grouping == "Group by Platforms and Media Buy Name":
        # Get unique combinations of Platforms and Media Buy Name
        platform_media_combinations = merged_df.dropna(subset=['Platforms', 'Media Buy Name']).groupby(['Platforms', 'Media Buy Name']).size().reset_index().drop(0, axis=1)
        
        for _, row in platform_media_combinations.iterrows():
            platform = row['Platforms']
            media_buy_name = row['Media Buy Name']
            filtered_df = merged_df[(merged_df['Platforms'] == platform) & (merged_df['Media Buy Name'] == media_buy_name)]
            display_performance(filtered_df, f"Platforms: {platform}, Media Buy Name: {media_buy_name}")
            insights = generate_insights(filtered_df, selected_kpi)
            st.write(f"### AI Insights for {platform}, {media_buy_name}")
            st.write(insights)



# Function to generate insights using Claude
def generate_insights(data, selected_kpi):
    focus_variables = ["Tags", "Asset Type", "Creative Content"]
    prompt = f"Analyze the following data focusing on {', '.join(focus_variables)} and the KPI {selected_kpi}. Determine which characteristics of the creatives work best. If the KPI starts with 'CP', best means the lowest value; otherwise, it means the highest value. The data is: {data.to_dict(orient='records')}"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    
    payload = {
        'prompt': prompt,
        'model': 'claude-3-7-sonnet-latest',
        'max_tokens': 2048,
        'temperature': 0.7,
    }
    
    response = requests.post('https://api.anthropic.com/v1/complete', headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["completion"]
    else:
        return "Failed to generate insights."

# Example usage in your app
if brandfolder_zip and brandfolder_csv and performance_data:
    # ... (rest of your code remains the same)
    
    # Create a dropdown for selecting a numeric KPI
    st.write("### Select a KPI to Analyze")
    selected_kpi = st.selectbox("Select a KPI", numeric_columns)

    # Create a dropdown for selecting grouping method
    st.write("### Select Performance Grouping")
    grouping_options = [
        "Aggregate (All Data)", 
        "Group by Platforms", 
        "Group by Media Buy Name", 
        "Group by Platforms and Media Buy Name"
    ]
    selected_grouping = st.selectbox("Select how to group the performance data", grouping_options)

    # Display performance based on selected grouping
    st.write("### Best and Worst Performing Creatives")
    
    if selected_grouping == "Aggregate (All Data)":
        insights = generate_insights(merged_df, selected_kpi)
        st.write("### AI Insights")
        st.write(insights)
        
    elif selected_grouping == "Group by Platforms":
        for platform in merged_df['Platforms'].dropna().unique():
            platform_df = merged_df[merged_df['Platforms'] == platform]
            insights = generate_insights(platform_df, selected_kpi)
            st.write(f"### AI Insights for {platform}")
            st.write(insights)
            
    elif selected_grouping == "Group by Media Buy Name":
        for media_buy_name in merged_df['Media Buy Name'].dropna().unique():
            media_buy_df = merged_df[merged_df['Media Buy Name'] == media_buy_name]
            insights = generate_insights(media_buy_df, selected_kpi)
            st.write(f"### AI Insights for {media_buy_name}")
            st.write(insights)
            
    elif selected_grouping == "Group by Platforms and Media Buy Name":
        # Get unique combinations of Platforms and Media Buy Name
        platform_media_combinations = merged_df.dropna(subset=['Platforms', 'Media Buy Name']).groupby(['Platforms', 'Media Buy Name']).size().reset_index().drop(0, axis=1)
        
        for _, row in platform_media_combinations.iterrows():
            platform = row['Platforms']
            media_buy_name = row['Media Buy Name']
            filtered_df = merged_df[(merged_df['Platforms'] == platform) & (merged_df['Media Buy Name'] == media_buy_name)]
            insights = generate_insights(filtered_df, selected_kpi)
            st.write(f"### AI Insights for {platform}, {media_buy_name}")
            st.write(insights)

#Next Steps
#1. Bring in the images
#2. Start looking for trends on what's working 
#3. Refine the trend analysis by controlling for key things.  Possibly choose how to control in a dropdown?