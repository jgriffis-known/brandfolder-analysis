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
MAX_IMAGE_WIDTH = 600
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
    brandfolder_zip = st.file_uploader("Brandfolder Zip", type=["zip"])
    brandfolder_csv = st.file_uploader("Brandfolder CSV", type=["csv"])
    performance_data = st.file_uploader("Performance Data XLSX", type=["xlsx"])

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

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith('.mov'):
                    mov_path = os.path.join(root, file)
                    mp4_path = mov_path.rsplit('.', 1)[0] + '.mp4'
                    try:
                        subprocess.run([
                            'ffmpeg',
                            '-i', mov_path,
                            '-vcodec', 'libx264',
                            '-acodec', 'aac',
                            mp4_path
                        ], check=True)
                        os.remove(mov_path)
                    except subprocess.CalledProcessError as e:
                        st.error(f"Error converting {file}: {str(e)}")

        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, 'w', zipfile.ZIP_DEFLATED) as new_zip:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, temp_dir)
                    new_zip.write(full_path, rel_path)
        mem_zip.seek(0)
        return mem_zip

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
                            return 'image', img_file.read()  # Return raw bytes instead of a Pillow object
                        except Exception as e:
                            print(f"Error loading image '{best_image_match}': {e}")

            # 2. Try to match video files
            video_files = [f for f in file_list if f.lower().endswith(valid_video_extensions)]
            if video_files:
                best_video_match = max(video_files, key=lambda f: fuzz.ratio(creative_name.lower(), os.path.basename(f).lower()))
                best_video_score = fuzz.ratio(creative_name.lower(), os.path.basename(best_video_match).lower())

                if best_video_score > 60:
                    with outer_zip.open(best_video_match) as video_file:
                        return 'video', video_file.read()

            return None, None

    except Exception as e:
        print(f"Zip handling error: {e}")
        return None, None

# Main app logic
if brandfolder_zip and brandfolder_csv and performance_data:
    # Data processing
    @st.cache_data
    def process_data():
        df_brandfolder = pd.read_csv(brandfolder_csv)
        df_performance = pd.read_excel(performance_data)
        
        numeric_cols = ['Media Cost', 'Impressions', 'Clicks']
        for col in numeric_cols:
            if col in df_performance.columns:
                try:
                    df_performance[col] = df_performance[col].apply(
                        lambda x: float(x.replace('$', '').replace(',', '')) if isinstance(x, str) else x
                    )
                except ValueError:
                    pass
        
        df_performance['Brandfolder Key'] = df_performance['Creative Name'].str.split('_').str[-1]
        return pd.merge(df_performance, df_brandfolder, left_on='Brandfolder Key', right_on='key', how='inner')

    merged_df = process_data()
    
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

    # Visualization section
    st.header("📈 Creative Performance")
    
    def display_creative_grid(df, title):
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
                            st.video(content)
                        else:
                            st.warning("No preview available")
                        st.metric(selected_kpi, f"{row[selected_kpi]:.2f}")

    # Display results based on grouping
    if selected_grouping == "Overall Performance":
        display_creative_grid(merged_df.nlargest(6, selected_kpi), "Top Performers")
        display_creative_grid(merged_df.nsmallest(6, selected_kpi), "Improvement Opportunities")
    else:
        # Add grouping-specific logic here
        pass

    # AI Insights
    st.header("🤖 AI Recommendations")
    with st.spinner("Generating insights..."):
        try:
            insights = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"Analyze these top performers ({selected_kpi}): {merged_df.nlargest(3, selected_kpi)['name'].tolist()}"
                }]
            ).content[0].text
            st.markdown(f"""
            <div style="
                padding: 1rem;
                border-radius: 0.5rem;
                background: {THEME_CONFIG['secondaryBackgroundColor']};
                margin: 1rem 0;
            ">
                {insights}
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to generate insights: {str(e)}")