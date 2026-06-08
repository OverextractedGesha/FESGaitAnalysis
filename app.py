import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

def manual_lpf(data_series, alpha):
    data = data_series.tolist()
    filtered = [data[0]] 
    for i in range(1, len(data)):
        val = alpha * data[i] + (1 - alpha) * filtered[-1]
        filtered.append(val)
    return filtered

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
        
    st.divider()
    st.header("Filter Settings")
    alpha_val = st.slider("LPF Alpha", min_value=0.01, max_value=1.0, value=0.15, step=0.01)

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        
        # Read the raw data without assuming column headers
        df = pd.read_csv(io.StringIO(content), sep=r'\s+', header=None)
        
        # Map specific column positions to their actual names
        # Assuming CH1-CH8 are the first 8 columns (index 0-7) 
        # and the Label is the very last column
        last_col_idx = len(df.columns) - 1
        rename_map = {
            0: 'L_Knee',
            1: 'L_Ankle',
            2: 'R_Knee',
            3: 'R_Ankle',
            4: 'L_Foot_Acc_Z',
            5: 'L_Foot_Acc_Y',
            6: 'R_Foot_Acc_Z',
            7: 'R_Foot_Acc_Y',
            last_col_idx: 'Label'
        }
        df.rename(columns=rename_map, inplace=True)

        # Generate a Time column (assumes 100Hz sampling rate)
        df['Time'] = np.arange(len(df)) * 0.01
        
        st.session_state['sensor_data'] = df
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

tab1, tab2 = st.tabs(["SENSOR SYSTEM (Gait Analysis)", "OPEN-LOOP FES SYSTEM"])

data_exists = isinstance(st.session_state.get('sensor_data'), pd.DataFrame)

with tab1:
    if data_exists:
        df = st.session_state['sensor_data']
        
        plot_col, param_col = st.columns([2, 1])
        
        with param_col:
            st.subheader("Parameters")
            st.info("Temporal parameter calculations require Heel/Toe sensor data, which is not present. Displaying full dataset.")
            
            st.markdown("**Knee Joint Parameters**")
            if 'L_Knee' in df.columns:
                st.metric("Max L_Knee Flexion [deg]", f"{df['L_Knee'].max():.1f}")
            if 'R_Knee' in df.columns:
                st.metric("Max R_Knee Flexion [deg]", f"{df['R_Knee'].max():.1f}")
                
            st.divider()
            st.markdown("**Ankle Joint Parameters**")
            if 'L_Ankle' in df.columns:
                st.metric("Max L_Ankle Dorsiflexion [deg]", f"{df['L_Ankle'].max():.1f}")
            if 'R_Ankle' in df.columns:
                st.metric("Max R_Ankle Dorsiflexion [deg]", f"{df['R_Ankle'].max():.1f}")

        with plot_col:
            st.subheader("Gait Analysis Plots (Full Record)")
            x_axis = df['Time']
            
            if 'L_Knee' in df.columns and 'R_Knee' in df.columns:
                fig_knee = go.Figure()
                fig_knee.add_trace(go.Scatter(x=x_axis, y=df['L_Knee'], name='Left Knee (Raw)', line=dict(color='lightgray')))
                fig_knee.add_trace(go.Scatter(x=x_axis, y=manual_lpf(df['L_Knee'], alpha_val), name='Left Knee (LPF)', line=dict(color='blue')))
                fig_knee.add_trace(go.Scatter(x=x_axis, y=manual_lpf(df['R_Knee'], alpha_val), name='Right Knee (LPF)', line=dict(color='orange', dash='dot')))
                fig_knee.update_layout(title="KNEE JOINT", xaxis_title="Time (s) [Assumed 100Hz]", yaxis_title="Deg", height=300, margin=dict(t=30, b=10))
                st.plotly_chart(fig_knee)

            if 'L_Ankle' in df.columns and 'R_Ankle' in df.columns:
                fig_ankle = go.Figure()
                fig_ankle.add_trace(go.Scatter(x=x_axis, y=df['L_Ankle'], name='Left Ankle (Raw)', line=dict(color='lightgray')))
                fig_ankle.add_trace(go.Scatter(x=x_axis, y=manual_lpf(df['L_Ankle'], alpha_val), name='Left Ankle (LPF)', line=dict(color='green')))
                fig_ankle.add_trace(go.Scatter(x=x_axis, y=manual_lpf(df['R_Ankle'], alpha_val), name='Right Ankle (LPF)', line=dict(color='purple', dash='dot')))
                fig_ankle.update_layout(title="ANKLE JOINT", xaxis_title="Time (s) [Assumed 100Hz]", yaxis_title="Deg", height=300, margin=dict(t=30, b=10))
                st.plotly_chart(fig_ankle)
                
            if 'Label' in df.columns:
                fig_label = go.Figure()
                fig_label.add_trace(go.Scatter(x=x_axis, y=df['Label'], name='Gait Label', line=dict(color='red', shape='hv')))
                fig_label.update_layout(title="GAIT PHASE (Label)", xaxis_title="Time (s) [Assumed 100Hz]", yaxis_title="Phase ID", height=200, margin=dict(t=30, b=10))
                st.plotly_chart(fig_label)
    else:
        st.info("Upload sensor data to view the Sensor System.")

with tab2:
    st.subheader("Open-Loop FES Configuration")
    fes_control_col, fes_plot_col = st.columns([1, 3])
    
    with fes_control_col:
        st.checkbox("1 Cycle")
        st.markdown("**Boost Properties**")
        st.number_input("Boost Thigh (V)", value=70)
        st.number_input("Boost Shank (V)", value=0)
        st.markdown("**Muscle Stimulation (ms)**")
        st.number_input("Periode Stim (s)", value=5)
        st.markdown("*Hip Flexion*")
        st.number_input("Iliopsoas", value=500)
        st.number_input("Rectus", value=500)
        st.markdown("*Knee Flexion*")
        st.number_input("BFLH", value=500)
        st.number_input("BFSH", value=500)
        st.number_input("Gastroc (KF)", value=500)
        st.button("START FES")
        st.button("STOP FES")

    with fes_plot_col:
        fig_boost = go.Figure()
        fig_boost.update_layout(title="Boost Voltage", height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig_boost)
        fig_fes_hip = go.Figure()
        fig_fes_hip.update_layout(title="HIP JOINT (FES Response)", height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig_fes_hip)