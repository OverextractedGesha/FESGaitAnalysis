import streamlit as st
import pandas as pd
import plotly.express as px
import io

st.set_page_config(page_title="Wearable FES-Gait System", layout="wide")
st.title("Wearable FES-Gait System Dashboard")

if 'sensor_data' not in st.session_state:
    st.session_state['sensor_data'] = None

with st.sidebar:
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Sensor Data (TXT)", type=['txt'])
    
    if st.button("Clear Data"):
        st.session_state['sensor_data'] = None
        st.rerun()

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        
        lines = content.split('\n')
        header_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('Time'):
                header_idx = i
                break
                
        df = pd.read_csv(io.StringIO(content), sep=r'\s+', skiprows=header_idx)
        st.session_state['sensor_data'] = df
        st.sidebar.success(f"Successfully loaded: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

if st.session_state['sensor_data'] is not None:
    df = st.session_state['sensor_data']
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.divider()
    
    st.subheader("Gait Analysis Plots")
    
    if 'HipKal' in df.columns:
        fig_hip = px.line(df, x='Time', y='HipKal', title="HIP JOINT")
        fig_hip.update_layout(xaxis_title="", yaxis_title="Deg", height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig_hip, use_container_width=True)

    if 'KneeKal' in df.columns:
        fig_knee = px.line(df, x='Time', y='KneeKal', title="KNEE JOINT")
        fig_knee.update_layout(xaxis_title="", yaxis_title="Deg", height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig_knee, use_container_width=True)

    if 'AnkleKal' in df.columns:
        fig_ankle = px.line(df, x='Time', y='AnkleKal', title="ANKLE JOINT")
        fig_ankle.update_layout(xaxis_title="Time", yaxis_title="Deg", height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig_ankle, use_container_width=True)

    gait_cols = [col for col in ['Heel', 'Toe'] if col in df.columns]
    if gait_cols:
        fig_gait = px.line(df, x='Time', y=gait_cols, title="GAIT PHASE")
        fig_gait.update_layout(xaxis_title="Time", yaxis_title="Volt", height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gait, use_container_width=True)