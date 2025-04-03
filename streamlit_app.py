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

# Streamlit app configuration
st.set_page_config(page_title="Brandfolder Creative Analysis", layout="wide")
st.title("🎨 Brandfolder Creative Analysis")

# Custom CSS for styling
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stDownloadButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    video {
        max-width: 800px !important;
    }
    img {
        max-width: 600px !important;
        height: auto !important;
    }
    </style>
""", unsafe_allow_html=True)

# File uploaders with instructions
with st.sidebar:
    st.header("📤 Upload Files")
    brandfolder_zip = st.file_uploader("Upload Brandfolder Zip", type=["zip"])
    brandfolder_csv = st.file_uploader("Upload Brandfolder CSV", type=["csv"])
    performance_data = st.file_uploader("Upload Performance Data XLSX", type=["xlsx"])

if brandfolder_zip and brandfolder_csv and performance_data:
    # Load data files
    df_brandfolder = pd.read_csv(brandfolder_csv)
    df_performance = pd.read_excel(performance_data)

    # Helper function to convert currency strings to float
    def convert_currency_to_float(x):
        if isinstance(x, str):
            return float(x.replace('$', '').replace(',', ''))
        return x

    # Apply conversion to numeric columns
    numeric_columns = ['Media Cost', 'Impressions', 'Clicks']
    for col in numeric_columns:
        if col in df_performance.columns:
            try:
                df_performance[col] = df_performance[col].apply(convert_currency_to_float)
            except ValueError:
                st.warning(f"Skipping column '{col}' as it cannot be converted to float.")

    # Merge dataframes based on Brandfolder Key
    df_performance['Brandfolder Key'] = df_performance['Creative Name'].str.split('_').str[-1]
    merged_df = pd.merge(df_performance, df_brandfolder, left_on='Brandfolder Key', right_on='key', how='inner')

    st.write("### Merged Data Preview")
    st.write(merged_df.head())

    # Provide download link for merged file as XLSX
    @st.cache_data
    def convert_df_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Merged Data')
        return output.getvalue()

    excel_data = convert_df_to_excel(merged_df)
    st.download_button("Download Merged XLSX", excel_data, "merged_file.xlsx")

    # Select KPI for analysis
    numeric_columns = merged_df.select_dtypes(include=['number']).columns.tolist()
    selected_kpi = st.selectbox("Select a KPI to Analyze", numeric_columns)

    # Grouping options dropdown
    grouping_options = ["Aggregate (All Data)", "Group by Platforms", "Group by Media Buy Name", "Group by Both"]
    selected_grouping = st.selectbox("Select Performance Grouping", grouping_options)

    # Function to find matching creative assets (images, videos, animated GIFs)
    def find_closest_matching_creative(creative_name, uploaded_zip):
        valid_image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
        valid_video_extensions = ('.mp4', '.mov', '.webm')

        try:
            with zipfile.ZipFile(uploaded_zip) as zip_ref:
                file_list = zip_ref.namelist()

                for file in file_list:
                    if 'animated' in file.lower() and file.lower().endswith(valid_image_extensions):
                        return 'animated_link', file  # Return link for animated GIFs
                    
                    if fuzz.ratio(creative_name.lower(), file.lower()) > 60:
                        if file.lower().endswith(valid_image_extensions):
                            with zip_ref.open(file) as img_file:
                                return 'image', Image.open(io.BytesIO(img_file.read()))
                        elif file.lower().endswith(valid_video_extensions):
                            with zip_ref.open(file) as video_file:
                                return 'video', video_file.read()

                return None, None

        except Exception as e:
            st.error(f"Error processing ZIP file: {e}")
            return None, None

    # Function to display creatives with proper handling for images and animated GIFs
    def display_creative(creative_name):
        match_type, content_or_link = find_closest_matching_creative(creative_name, brandfolder_zip)

        if match_type == 'image':
            st.image(content_or_link, caption=creative_name)
        elif match_type == 'animated_link':
            st.markdown(f"[View Animated GIF: {creative_name}]({content_or_link})")
        elif match_type == 'video':
            st.video(content_or_link)
        else:
            st.warning(f"No preview available for {creative_name}.")

    # Display performance based on grouping selection
    def display_performance(df, title):
        st.subheader(title)
        
        for _, row in df.iterrows():
            creative_name = row["Creative Name"]
            display_creative(creative_name)
            st.metric(selected_kpi, f"{row[selected_kpi]:.2f}")

    # Show results based on grouping method
    if selected_grouping == "Aggregate (All Data)":
        display_performance(merged_df.nlargest(6, selected_kpi), "Top Performers")
        display_performance(merged_df.nsmallest(6, selected_kpi), "Improvement Opportunities")