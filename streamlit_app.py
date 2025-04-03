"""Main application file for Brandfolder Creative Analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import tempfile
import subprocess
import os
import re
import shutil
from pathlib import Path
from PIL import Image
from anthropic import Anthropic
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from io import BytesIO

# Configuration
load_dotenv()
MAX_IMAGE_WIDTH = 600
MAX_VIDEO_WIDTH = 800
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
    video {{
        max-width: {MAX_VIDEO_WIDTH}px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ... [Keep all helper functions identical] ...

# File upload section
with st.sidebar:
    st.header("📤 Upload Files")
    brandfolder_zip = st.file_uploader("Brandfolder Zip", type=["zip"])
    brandfolder_csv = st.file_uploader("Brandfolder CSV", type=["csv"])
    performance_data = st.file_uploader("Performance Data XLSX", type=["xlsx"])

if brandfolder_zip and brandfolder_csv and performance_data:
    # Enhanced data processing with validation
    @st.cache_data
    def process_data():
        try:
            df_brandfolder = pd.read_csv(brandfolder_csv)
            df_performance = pd.read_excel(performance_data)
            
            # Validate required columns exist
            required_columns = ['Platform', 'Media Buy Name', 'Creative Name']
            missing_cols = [col for col in required_columns if col not in df_performance.columns]
            if missing_cols:
                st.error(f"Missing required columns in performance data: {', '.join(missing_cols)}")
                st.stop()

            # Clean numeric columns
            numeric_cols = ['Media Cost', 'Impressions', 'Clicks']
            for col in numeric_cols:
                if col in df_performance.columns:
                    df_performance[col] = df_performance[col].replace('[\$,]', '', regex=True).astype(float)
            
            # Extract Brandfolder Key and merge
            df_performance['Brandfolder Key'] = df_performance['Creative Name'].str.extract(r'_([^_]+)$')
            merged = pd.merge(df_performance, df_brandfolder, 
                            left_on='Brandfolder Key', 
                            right_on='key', 
                            how='inner')

            # Show merged data preview
            st.subheader("Merged Data Preview")
            st.dataframe(merged.head())
            
            # Add download button for merged data
            @st.cache_data
            def convert_df_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False)
                return output.getvalue()
            
            excel_data = convert_df_to_excel(merged)
            st.download_button(
                label="Download Merged Data (XLSX)",
                data=excel_data,
                file_name="merged_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            return merged
            
        except Exception as e:
            st.error(f"Data processing error: {str(e)}")
            st.stop()

    merged_df = process_data()
    brandfolder_zip = convert_mov_to_mp4(brandfolder_zip)

    # Analysis controls
    st.header("📊 Performance Analysis")
    col1, col2 = st.columns(2)
    with col1:
        selected_kpi = st.selectbox("Select KPI", merged_df.select_dtypes(include=np.number).columns.tolist())
    with col2:
        selected_grouping = st.selectbox("Group By", [
            "Overall Performance", 
            "By Platform", 
            "By Media Buy", 
            "Platform & Media Buy"
        ])

    # Visualization
    def display_creative_group(df, title):
        with st.expander(title):
            cols = st.columns(3)
            for idx, (_, row) in enumerate(df.iterrows()):
                with cols[idx % 3]:
                    with st.container():
                        st.caption(f"Creative: {row['name']}")
                        match_type, content = find_closest_matching_creative(row["name"], brandfolder_zip)
                        if match_type == "image":
                            display_image(content, "")
                        elif match_type == "video":
                            handle_video(content, row["name"])
                        else:
                            st.warning("No preview available")
                        st.metric(selected_kpi, f"{row[selected_kpi]:.2f}")

    # Grouping logic
    st.header("📈 Creative Performance")
    try:
        if selected_grouping == "Overall Performance":
            display_creative_group(merged_df.nlargest(6, selected_kpi), "Top Performers")
            display_creative_group(merged_df.nsmallest(6, selected_kpi), "Improvement Opportunities")
            
        elif selected_grouping == "By Platform":
            for platform in merged_df['Platform'].unique():
                platform_df = merged_df[merged_df['Platform'] == platform]
                display_creative_group(platform_df.nlargest(3, selected_kpi), f"Top Performers - {platform}")
                display_creative_group(platform_df.nsmallest(3, selected_kpi), f"Improvement Opportunities - {platform}")
                
        elif selected_grouping == "By Media Buy":
            for media_buy in merged_df['Media Buy Name'].unique():
                media_buy_df = merged_df[merged_df['Media Buy Name'] == media_buy]
                display_creative_group(media_buy_df.nlargest(3, selected_kpi), f"Top Performers - {media_buy}")
                display_creative_group(media_buy_df.nsmallest(3, selected_kpi), f"Improvement Opportunities - {media_buy}")
                
        elif selected_grouping == "Platform & Media Buy":
            grouped = merged_df.groupby(['Platform', 'Media Buy Name'])
            for (platform, media_buy), group_df in grouped:
                if not group_df.empty:
                    display_creative_group(group_df.nlargest(2, selected_kpi), f"Top Performers - {platform} | {media_buy}")
                    display_creative_group(group_df.nsmallest(2, selected_kpi), f"Improvement Opportunities - {platform} | {media_buy}")

        # AI Recommendations
        st.header("🤖 AI Recommendations")
        with st.spinner("Generating insights..."):
            try:
                insights = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": f"Analyze these {len(merged_df)} creatives based on {selected_kpi}. " +
                                   "Identify top 3 success factors and improvement opportunities. " +
                                   "Focus on visual elements, messaging, and technical specs. " +
                                   "Format with emojis and bullet points."
                    }]
                ).content[0].text
                
                st.markdown(f"""
                <div style="
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    background: {THEME_CONFIG['secondaryBackgroundColor']};
                    margin: 1rem 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <h4 style='color:{THEME_CONFIG['primaryColor']}; margin-top:0;'>🎯 Key Recommendations</h4>
                    {insights}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Failed to generate insights: {str(e)}")
                
    except KeyError as e:
        st.error(f"Data validation error: Missing required column - {str(e)}")
        st.error("Please check your input files contain 'Platform' and 'Media Buy Name' columns")
        
else:
    st.warning("Please upload all required files to proceed.")