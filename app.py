import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Wearable FES-Gait System", layout="wide")
st.title("Wearable FES-Gait System Dashboard")

# Initialize session state
if 'sensor_data' not in st.session_state:
    st.session_state['sensor_data'] = None

with st.sidebar:
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Sensor Data (TXT)", type=['txt'])
    
    if st.button("Clear Data"):
        st.session_state['sensor_data'] = None
        st.rerun()

# Data Parsing
if uploaded_file is not None:
    try:
        # Skip the first 3 rows (metadata) and parse using whitespace delimiter
        df = pd.read_csv(uploaded_file, sep=r'\s+', skiprows=3)
        st.session_state['sensor_data'] = df
        st.sidebar.success(f"Successfully loaded: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

# Data Visualization
if st.session_state['sensor_data'] is not None:
    df = st.session_state['sensor_data']
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.divider()
    
    st.subheader("Gait Analysis Visualization")
    
    # Auto-select the calculated Kalman joint angles if they exist in the file
    default_y_axes = [col for col in ['HipKal', 'KneeKal', 'AnkleKal'] if col in df.columns]
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-Axis:", df.columns, index=0)
    with col2:
        y_axes = st.multiselect("Select Y-Axes:", df.columns, default=default_y_axes)
        
    if y_axes:
        fig = px.line(
            df, 
            x=x_axis, 
            y=y_axes, 
            title="Joint Angles over Time",
            labels={"value": "Degrees", "variable": "Joint"}
        )
        # Force the layout to mimic the zero-centered appearance in your Delphi screenshots
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload a Gait Sensor Data file to begin analysis.")