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

    # Function to match creative name to image or video or animated GIFs
    def find_closest_matching_creative(creative_name, uploaded_zip):
        valid_image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
        valid_video_extensions = ('.mp4', '.mov', '.webm')

        try:
            with zipfile.ZipFile(uploaded_zip) as outer_zip:
                file_list = outer_zip.namelist()

                # Match images or videos
                for file in file_list:
                    if 'animated' in file.lower() and file.lower().endswith(valid_image_extensions):
                        return 'animated_link', file  # Return link for animated GIFs
                    
                    if fuzz.ratio(creative_name.lower(), file.lower()) > 60:
                        if file.lower().endswith(valid_image_extensions):
                            with outer_zip.open(file) as img_file:
                                return 'image', Image.open(io.BytesIO(img_file.read()))
                        elif file.lower().endswith(valid_video_extensions):
                            with outer_zip.open(file) as video_file:
                                return 'video', video_file.read()

                return None, None

        except Exception as e:
            print(f"Zip handling error: {e}")
            return None, None
    
    # Function to display performance for a group
    def display_performance(df, group_name=""):
        if df.empty or selected_kpi not in df.columns or df[selected_kpi].isna().all():
            st.write(f"**{group_name}**")
            st.write("No valid data available for this group.")
            return
        
        grouped_by_creative = df.groupby('Creative Name')
        
        for creative_name, creative_data in grouped_by_creative:
            match_type, content_or_link = find_closest_matching_creative(creative_name, brandfolder_zip)
            
            if match_type == 'image':
                st.image(content_or_link, caption=creative_name, width=300)  # Limit width of static images
            
            elif match_type == 'animated_link':
                st.markdown(f"[View Animated GIF: {creative_name}]({content_or_link})")
            
            elif match_type == 'video':
                st.video(content_or_link)
            
            else:
                st.warning(f"No preview available for {creative_name}.")
    
    # Display performance based on selected grouping
    if selected_grouping == "Aggregate (All Data)":
        display_performance(merged_df, "All Data")