import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Wearable FES-Gait System", layout="wide")
st.title("Wearable FES-Gait System Dashboard")

if 'sensor_data' not in st.session_state:
    st.session_state['sensor_data'] = None

with st.sidebar:
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Sensor Data (TXT/CSV)", type=['txt', 'csv'])
    
    if st.button("Clear Data"):
        st.session_state['sensor_data'] = None
        st.rerun()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    st.session_state['sensor_data'] = df

if st.session_state['sensor_data'] is not None:
    df = st.session_state['sensor_data']
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.divider()
    
    st.subheader("Gait Analysis Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-Axis:", df.columns)
    with col2:
        y_axes = st.multiselect("Select Y-Axes:", df.columns)
        
    if y_axes:
        fig = px.line(df, x=x_axis, y=y_axes, title="Joint Angles", labels={"value": "Degrees", "variable": "Joint"})
        st.plotly_chart(fig, use_container_width=True)