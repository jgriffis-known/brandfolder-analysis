"""Main application file for Brandfolder Creative Analysis."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
st.sidebar.markdown("Utilize Pivot Tables in MCI with the following variables: Platform, Campaign Name, Media Buy (i.e. Audience), Creative Name, Media Cost, Impressions, Clicks, and any other KPI you'd like to use.")
st.sidebar.markdown("Creative Name must have the brandfolder key in the naming convention.")


if brandfolder_zip and brandfolder_csv and performance_data:
    df_brandfolder = pd.read_csv(brandfolder_csv)
    df_performance = pd.read_excel(performance_data)
    
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
    st.write("### Choose a KPI for analysis")
    selected_kpi = st.selectbox("Select a KPI", numeric_columns)


# Extract unique media buy names from the merged DataFrame
unique_media_buy_names = merged_df['Media Buy Name'].unique()

# Determine best performing creative for each media buy name
st.write("### Best Performing Creatives by Media Buy Name")
for media_buy_name in unique_media_buy_names:
    media_buy_df = merged_df[merged_df['Media Buy Name'] == media_buy_name]
    
    if selected_kpi.startswith('CP'):
        # For KPIs starting with 'CP', use the lowest value
        best_performing_creative = media_buy_df.loc[media_buy_df[selected_kpi].idxmin()]
    else:
        # For other KPIs, use the highest value
        best_performing_creative = media_buy_df.loc[media_buy_df[selected_kpi].idxmax()]
    
    st.write(f"**Media Buy Name:** {media_buy_name}")
    st.write(f"**Best Performing Creative:** {best_performing_creative['name']}")
    st.write(f"**KPI Value:** {best_performing_creative[selected_kpi]}")
    st.write("---")

###steps id like to take next
#####might require auto-changing any $ variables to numeric first
##2. analysis boxes should be done by campaign, by audience, and then overall

