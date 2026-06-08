import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

# --- Manual LPF Implementation ---
def manual_lpf(data_series, alpha):
    data = data_series.tolist()
    filtered = [data[0]] 
    for i in range(1, len(data)):
        val = alpha * data[i] + (1 - alpha) * filtered[-1]
        filtered.append(val)
    return filtered

# --- Page Config ---
st.set_page_config(page_title="Wearable FES-Gait System", layout="wide")
st.title("Wearable FES-Gait System Graphical User Interface")

# --- Session State ---
if 'sensor_data' not in st.session_state:
    st.session_state['sensor_data'] = None

# --- Sidebar: Data Management ---
with st.sidebar:
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Sensor Data (TXT)", type=['txt'])
    
    if st.button("Clear Data"):
        st.session_state['sensor_data'] = None
        st.rerun()
        
    st.divider()
    st.header("Filter Settings")
    alpha_val = st.slider("LPF Alpha", min_value=0.01, max_value=1.0, value=0.15, step=0.01)

# --- File Processing ---
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
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

# --- Main Dashboard Tabs ---
tab1, tab2 = st.tabs(["SENSOR SYSTEM (Gait Analysis)", "OPEN-LOOP FES SYSTEM"])

# ==========================================
# TAB 1: SENSOR SYSTEM
# ==========================================
with tab1:
    if st.session_state['sensor_data'] is not None:
        df = st.session_state['sensor_data']
        
        plot_col, param_col = st.columns([2, 1])
        
        with plot_col:
            st.subheader("Gait Analysis Plots")
            
            # Hip Plot
            if 'HipKal' in df.columns:
                fig_hip = go.Figure()
                fig_hip.add_trace(go.Scatter(x=df['Time'], y=df['HipKal'], name='Raw', line=dict(color='lightgray')))
                fig_hip.add_trace(go.Scatter(x=df['Time'], y=manual_lpf(df['HipKal'], alpha_val), name='LPF', line=dict(color='red')))
                fig_hip.update_layout(title="HIP JOINT", height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_hip, use_container_width=True)

            # Knee Plot
            if 'KneeKal' in df.columns:
                fig_knee = go.Figure()
                fig_knee.add_trace(go.Scatter(x=df['Time'], y=df['KneeKal'], name='Raw', line=dict(color='lightgray')))
                fig_knee.add_trace(go.Scatter(x=df['Time'], y=manual_lpf(df['KneeKal'], alpha_val), name='LPF', line=dict(color='blue')))
                fig_knee.update_layout(title="KNEE JOINT", height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_knee, use_container_width=True)

            # Ankle Plot
            if 'AnkleKal' in df.columns:
                fig_ankle = go.Figure()
                fig_ankle.add_trace(go.Scatter(x=df['Time'], y=df['AnkleKal'], name='Raw', line=dict(color='lightgray')))
                fig_ankle.add_trace(go.Scatter(x=df['Time'], y=manual_lpf(df['AnkleKal'], alpha_val), name='LPF', line=dict(color='green')))
                fig_ankle.update_layout(title="ANKLE JOINT", height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_ankle, use_container_width=True)

        with param_col:
            st.subheader("Parameters")
            st.selectbox("CYCLE", ["Siklus ke - 1", "Siklus ke - 2", "Siklus ke - 3"])
            
            st.markdown("**Temporal Parameters**")
            col_t1, col_t2 = st.columns(2)
            col_t1.metric("IC [%time]", "0.0")
            col_t1.metric("FF [%time]", "0.0")
            col_t1.metric("HO [%time]", "0.0")
            col_t2.metric("Tstance [%time]", "0.0")
            col_t2.metric("Tswing [%time]", "0.0")
            col_t2.metric("Cad [strd/min]", "0.0")
            
            st.divider()
            st.markdown("**Hip Joint Parameters**")
            st.metric("HIC [deg]", "0.0")
            st.metric("MHEst [deg]", "0.0")
            
            st.divider()
            st.markdown("**Knee Joint Parameters**")
            st.metric("KIC [deg]", "0.0")
            st.metric("MKFst [deg]", "0.0")
    else:
        st.info("Upload sensor data to view the Sensor System.")

# ==========================================
# TAB 2: OPEN-LOOP FES SYSTEM
# ==========================================
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
        # Placeholder plots for FES output
        st.info("FES Simulation Plots will render here based on the stimulation parameters on the left.")
        
        fig_boost = go.Figure()
        fig_boost.update_layout(title="Boost Voltage", height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig_boost, use_container_width=True)
        
        fig_fes_hip = go.Figure()
        fig_fes_hip.update_layout(title="HIP JOINT (FES Response)", height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig_fes_hip, use_container_width=True)