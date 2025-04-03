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

# Custom CSS for styling metrics and layout adjustments
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
    .stMetric {{
        font-size: 0.9rem !important; /* Adjust font size for metrics */
        overflow-wrap: break-word;   /* Ensure long numbers wrap */
    }}
    </style>
    """, unsafe_allow_html=True)

def sanitize_filename(filename):
    """Remove invalid characters from filenames"""
    return re.sub(r'[\\/*?:"<>|\x00]', "", str(filename))

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

def handle_video(content, creative_name):
    """Handle video display with proper temp file management"""
    try:
        temp_dir = Path(tempfile.mkdtemp())
        temp_file = temp_dir / f"{sanitize_filename(creative_name)}.mp4"
        
        with open(temp_file, "wb") as f:
            f.write(content)
        
        st.video(str(temp_file), format="video/mp4", start_time=0)
    except Exception as e:
        st.error(f"Error displaying video: {str(e)}")

def convert_mov_to_mp4(zip_file):
    """Convert .mov files to .mp4 with enhanced error handling"""
    if not shutil.which('ffmpeg'):
        st.error("FFmpeg not found! Please install FFmpeg and ensure it's in your PATH.")
        return zip_file

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith('.mov'):
                        mov_path = Path(root) / file
                        mp4_path = mov_path.with_suffix('.mp4')
                        
                        try:
                            subprocess.run([
                                'ffmpeg', '-y',
                                '-i', str(mov_path),
                                '-c:v', 'libx264',
                                '-preset', 'fast',
                                '-crf', '22',
                                '-c:a', 'aac',
                                '-b:a', '128k',
                                '-pix_fmt', 'yuv420p',
                                str(mp4_path)
                            ], check=True, capture_output=True)
                            mov_path.unlink()
                        except subprocess.CalledProcessError as e:
                            st.error(f"Error converting {file}: {e.stderr.decode()}")

            mem_zip = BytesIO()
            with zipfile.ZipFile(mem_zip, 'w', zipfile.ZIP_DEFLATED) as new_zip:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        full_path = Path(root) / file
                        rel_path = full_path.relative_to(temp_dir)
                        new_zip.write(full_path, rel_path.as_posix())
            
            mem_zip.seek(0)
            return mem_zip
    except Exception as e:
        st.error(f"Video conversion error: {str(e)}")
        return zip_file

def find_closest_matching_creative(creative_name, uploaded_zip):
    """Match creative name to assets with improved error handling"""
    valid_image_ext = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
    valid_video_ext = ('.mp4', '.mov', '.webm')
    
    try:
        with zipfile.ZipFile(uploaded_zip) as zip_ref:
            file_list = [sanitize_filename(f) for f in zip_ref.namelist()]

            # Try matching images first
            image_files = [f for f in file_list if f.lower().endswith(valid_image_ext)]
            if image_files:
                best_image = max(image_files, key=lambda f: fuzz.ratio(creative_name.lower(), Path(f).stem.lower()))
                if fuzz.ratio(creative_name.lower(), Path(best_image).stem.lower()) > 60:
                    with zip_ref.open(best_image) as f:
                        return 'image', BytesIO(f.read())

            # Try matching videos
            video_files = [f for f in file_list if f.lower().endswith(valid_video_ext)]
            if video_files:
                best_video = max(video_files, key=lambda f: fuzz.ratio(creative_name.lower(), Path(f).stem.lower()))
                if fuzz.ratio(creative_name.lower(), Path(best_video).stem.lower()) > 60:
                    with zip_ref.open(best_video) as f:
                        return 'video', f.read()

            return None, None
    except Exception as e:
        st.error(f"File processing error: {str(e)}")
        return None, None

# File upload section
with st.sidebar:
    st.header("📤 Upload Files")
    brandfolder_zip = st.file_uploader("Brandfolder Zip", type=["zip"])
    brandfolder_csv = st.file_uploader("Brandfolder CSV", type=["csv"])
    performance_data = st.file_uploader("Performance Data XLSX", type=["xlsx"])

if brandfolder_zip and brandfolder_csv and performance_data:
    
    @st.cache_data
    def process_data():
        df_brandfolder = pd.read_csv(brandfolder_csv)
        df_performance = pd.read_excel(performance_data)
        
        numeric_cols = ['Media Cost', 'Impressions', 'Clicks']
        for col in numeric_cols:
            if col in df_performance.columns:
                try:
                    df_performance[col] = df_performance[col].replace('[\$,]', '', regex=True).astype(float)
                except Exception as e:
                    st.error(f"Error converting {col}: {str(e)}")
        
        df_performance['Brandfolder Key'] = df_performance['Creative Name'].str.extract(r'_([^_]+)$')
        return pd.merge(df_performance, df_brandfolder, left_on='Brandfolder Key', right_on='key', how='inner')

    merged_df = process_data()
    
    brandfolder_zip = convert_mov_to_mp4(brandfolder_zip)
    
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
                        elif "Animated" in row["name"] or row["name"].lower().endswith(".gif"):
                            # Display a link for animated GIFs instead of warning.
                            sanitized_name = sanitize_filename(row["name"])
                            st.markdown(f"[Download Animated GIF]({sanitized_name})")
                        else:
                            st.warning("No preview available")
                        
                        # Display metric value.
                        st.metric(selected_kpi, f"{row[selected_kpi]:.2f}")

    
    if selected_grouping == "Overall Performance":
        