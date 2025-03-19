"""Main application file for Brandfolder Creative Analysis."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from anthropic import Anthropic
import requests

# Set up your API key
api_key = "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"
client = Anthropic(api_key=api_key)

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

    # Apply the conversion function to all columns in df_performance
    for col in df_performance.columns:
        if df_performance[col].dtype == 'object':  # Check if column is of object type
            try:
                df_performance[col] = df_performance[col].apply(convert_currency_to_float)
            except ValueError:
                # Handle columns that cannot be converted to float
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
        
        st.write("#### Worst Performing Creatives:")
        for index, row in worst_performers.iterrows():
            metric_value = row[selected_kpi]
            if pd.notnull(metric_value):
                formatted_value = f"{metric_value:.2f}" if isinstance(metric_value, (float, int)) else metric_value
                st.write(f"- **Creative Name:** {row['name']}, **{selected_kpi}:** {formatted_value}")
        
        st.write("---")

    # Display performance based on selected grouping
    st.write("### Best and Worst Performing Creatives")
    
    if selected_grouping == "Aggregate (All Data)":
        display_performance(merged_df, "All Data")
        
    elif selected_grouping == "Group by Platforms":
        for platform in merged_df['Platforms'].dropna().unique():
            platform_df = merged_df[merged_df['Platforms'] == platform]
            display_performance(platform_df, f"Platforms: {platform}")
            
    elif selected_grouping == "Group by Media Buy Name":
        for media_buy_name in merged_df['Media Buy Name'].dropna().unique():
            media_buy_df = merged_df[merged_df['Media Buy Name'] == media_buy_name]
            display_performance(media_buy_df, f"Media Buy Name: {media_buy_name}")
            
    elif selected_grouping == "Group by Platforms and Media Buy Name":
        # Get unique combinations of Platforms and Media Buy Name
        platform_media_combinations = merged_df.dropna(subset=['Platforms', 'Media Buy Name']).groupby(['Platforms', 'Media Buy Name']).size().reset_index().drop(0, axis=1)
        
        for _, row in platform_media_combinations.iterrows():
            platform = row['Platforms']
            media_buy_name = row['Media Buy Name']
            filtered_df = merged_df[(merged_df['Platforms'] == platform) & (merged_df['Media Buy Name'] == media_buy_name)]
            display_performance(filtered_df, f"Platforms: {platform}, Media Buy Name: {media_buy_name}")



# Function to generate insights using Claude
def generate_insights(data, selected_kpi, focus_variables):
    api_key = "YOUR_API_KEY_HERE"
    url = f"https://api.anthropic.com/v1/complete"
    
    prompt = f"Analyze the following data focusing on {', '.join(focus_variables)} and the KPI {selected_kpi}. Determine which characteristics of the creatives work best. If the KPI starts with 'CP', best means the lowest value; otherwise, it means the highest value. The data is: {data.to_dict(orient='records')}"
    
    payload = {
        "prompt": prompt,
        "model": "claude-3-7-sonnet-latest",
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["completion"]
    else:
        return "Failed to generate insights."

# Example usage
if brandfolder_zip and brandfolder_csv and performance_data:
    # ... (rest of your code remains the same)
    
    # Extract numeric columns from df_performance
    numeric_columns = df_performance.select_dtypes(include=['number']).columns.tolist()

    # Create a dropdown for selecting a numeric KPI
    st.write("### Select a KPI to Analyze")
    selected_kpi = st.selectbox("Select a KPI", numeric_columns)

    # Focus variables for Claude
    focus_variables = ["Tags", "Asset Type", "Creative Content"]

    # Generate insights using Claude
    insights = generate_insights(merged_df, selected_kpi, focus_variables)

    st.write("### AI Insights")
    st.write(insights)

#Next Steps
#1. Bring in the images
#2. Start looking for trends on what's working 
#3. Refine the trend analysis by controlling for key things.  Possibly choose how to control in a dropdown?