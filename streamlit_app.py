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
import os
import requests
from io import BytesIO
from dotenv import load_dotenv

# Configuration
load_dotenv()
MAX_IMAGE_WIDTH = 600  # Adjust based on your preference
THEME_CONFIG = {
    "primaryColor": "#4CAF50",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F0F0F0",
    "textColor": "#000000",
    "font": "sans serif"
}

# Initialize Claude client
client = Anthropic(api_key=os.getenv('api_key'))

# Streamlit app configuration
st.set_page_config(page_title="Brandfolder Analytics", layout="wide")
st.title("🎨 Brandfolder Creative Analysis")

# Custom CSS for styling
st.markdown(f"""
    <style>
    .reportview-container .main .block-container{{
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    .stDownloadButton > button {{
        background-color: {THEME_CONFIG['primaryColor']};
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# File upload section
with st.sidebar:
    st.header("📤 Upload Files")
    
    brandfolder_zip = st.file_uploader("Brandfolder Zip", type=["zip"],
                                      help="Select all files you'd like to include in Brandfolder and download as zip")
    brandfolder_csv = st.file_uploader("Brandfolder CSV", type=["csv"],
                                      help="Select all files you'd like to include in Brandfolder and download as CSV")
    performance_data = st.file_uploader("Performance Data XLSX", type=["xlsx"],
                                       help="Utilize Pivot Tables in MCI with required variables")

def display_image(content, caption):
    """Display image with controlled dimensions"""
    try:
        img = Image.open(content)
        wpercent = (MAX_IMAGE_WIDTH / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((MAX_IMAGE_WIDTH, hsize), Image.Resampling.LANCZOS)
        st.image(img, caption=caption, use_column_width=False)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def convert_mov_to_mp4(zip_file):
    """Convert .mov files to .mp4 with error handling"""
    if not shutil.which('ffmpeg'):
        st.error("FFmpeg not found! Please install FFmpeg and ensure it's in your PATH.")
        return zip_file

    # ... rest of conversion function remains same ...

def generate_insights(data, selected_kpi):
    """Generate AI insights with improved error handling"""
    focus_vars = ["Tags", "Asset Type", "Creative Content"]
    prompt = f"""Analyze this marketing creative data focusing on {', '.join(focus_vars)} and KPI {selected_kpi}.
    Identify top performing characteristics. Format response with bullet points. 
    Data: {data.to_dict(orient='records')}"""
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"AI Insight Generation Failed: {str(e)}")
        return "Insights unavailable - please try again later."

# Main app logic
if brandfolder_zip and brandfolder_csv and performance_data:
    # Data processing (cached)
    @st.cache_data
    def process_data():
        # ... data processing logic remains same ...
        return merged_df
    
    merged_df = process_data()
    
    # KPI Selection
    st.header("📊 Performance Analysis")
    numeric_cols = merged_df.select_dtypes(include=np.number).columns.tolist()
    selected_kpi = st.selectbox("Select Key Performance Indicator", numeric_cols)
    
    # Grouping Selection
    grouping = st.selectbox("Analysis Grouping", [
        "Overall Performance", 
        "By Platform", 
        "By Media Buy", 
        "Platform & Media Buy Combination"
    ])
 
    # Display creative content in grid
    # Visualization Section
    st.header("📈 Creative Performance")
    with st.expander("Advanced Filters"):
        # Add filter controls here
        
    # Display creative content in grid
    def display_creative_grid(df, title):  # Now properly indented
        with st.container():
            st.subheader(title)
            cols = st.columns(3)
            for idx, (_, row) in enumerate(df.iterrows()):
                with cols[idx % 3]:
                    with st.container():
                        st.caption(f"Creative: {row['name']}")
                        match_type, content = find_closest_matching_creative(row["name"], brandfolder_zip)
                        
                        if match_type == "image":
                            display_image(content, "")
                        elif match_type == "video":
                            st.video(content)
                        elif match_type == "html_link":
                            st.markdown(f"[Interactive Creative]({content})", unsafe_allow_html=True)
                        else:
                            st.warning("No preview available")
                            
                        st.metric(selected_kpi, f"{row[selected_kpi]:.2f}")

    # Display AI Insights in formatted box            
    def display_insights(insights_text):
        st.markdown(f"""
        <div style="
            padding: 1rem;
            border-radius: 0.5rem;
            background: {THEME_CONFIG['secondaryBackgroundColor']};
            margin: 1rem 0;
        ">
            <h4 style='color:{THEME_CONFIG['primaryColor']};'>AI Recommendations</h4>
            {insights_text}
        </div>
        """, unsafe_allow_html=True)

    # Main analysis flow
    if grouping == "Overall Performance":
        display_creative_grid(merged_df.nlargest(6, selected_kpi), "Top Performers")
        display_creative_grid(merged_df.nsmallest(6, selected_kpi), "Improvement Opportunities")
        display_insights(generate_insights(merged_df, selected_kpi))
    
    # ... other grouping cases ...

# Requirements: Create requirements.txt with
# streamlit anthropic python-dotenv pillow fuzzywumpy pandas openpyxl xlsxwriter