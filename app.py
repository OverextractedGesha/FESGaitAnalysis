import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io

def manual_lpf(data_series, alpha):
    # Convert to list and handle potential string/NaN values gracefully
    data = pd.to_numeric(data_series, errors='coerce').fillna(0).tolist()
    if not data: return []
    
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
        
        # Read raw data, allowing pandas to handle multiple spaces
        df = pd.read_csv(io.StringIO(content), sep=r'\s+', header=None)
        
        # Drop any empty columns that might have been created by trailing spaces
        df = df.dropna(axis=1, how='all')
        
        # Check if the file has a header row (if first item in Time is a string, skip row 0)
        if isinstance(df.iloc[0, 0], str) and not df.iloc[0, 0].replace('.','',1).isdigit():
            df = df.iloc[1:].reset_index(drop=True)
            
        # Ensure we only apply this to exactly 15 columns
        if len(df.columns) >= 15:
            # Take only the first 15 columns in case of trailing delimiters
            df = df.iloc[:, :15] 
            df.columns = [
                'Time', 'Heel', 'Toe', 'Hip', 'Knee', 'Ankle', 
                'Gluteus_Maximus', 'Biceps_Femoris_Short', 'Biceps_Femoris_Long', 
                'Vastus_Medialis', 'Vastus_Lateralis', 'Rectus_Femoris', 
                'Soleus', 'Gastrocnemius', 'Tibialis_Anterior'
            ]
            
            # Convert all columns to numeric, forcing any weird text to NaN, then to 0
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            st.session_state['sensor_data'] = df
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        else:
            st.sidebar.error(f"Data format mismatch. Found {len(df.columns)} columns, expected 15.")
            
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

tab1, tab2 = st.tabs(["SENSOR SYSTEM (Gait Analysis)", "OPEN-LOOP FES SYSTEM"])

data_exists = isinstance(st.session_state.get('sensor_data'), pd.DataFrame)

with tab1:
    if data_exists:
        df = st.session_state['sensor_data']
        
        plot_col, param_col = st.columns([2, 1])
        
        with param_col:
            st.subheader("Joint Parameters")
            st.metric("Max Hip Angle [deg]", f"{df['Hip'].max():.1f}")
            st.metric("Max Knee Angle [deg]", f"{df['Knee'].max():.1f}")
            st.metric("Max Ankle Angle [deg]", f"{df['Ankle'].max():.1f}")

        with plot_col:
            st.subheader("Kinematic Joint Plots")
            x_axis = df['Time']
            
            # 1. HIP PLOT
            fig_hip = go.Figure()
            fig_hip.add_trace(go.Scatter(x=x_axis, y=df['Hip'], name='Raw', line=dict(color='lightgray')))
            fig_hip.add_trace(go.Scatter(x=x_axis, y=manual_lpf(df['Hip'], alpha_val), name='LPF', line=dict(color='red')))
            fig_hip.update_layout(title="HIP JOINT", xaxis_title="Time (s)", yaxis_title="Angle (Deg)", height=250, margin=dict(t=30, b=10))
            st.plotly_chart(fig_hip)

            # 2. KNEE PLOT
            fig_knee = go.Figure()
            fig_knee.add_trace(go.Scatter(x=x_axis, y=df['Knee'], name='Raw', line=dict(color='lightgray')))
            fig_knee.add_trace(go.Scatter(x=x_axis, y=manual_lpf(df['Knee'], alpha_val), name='LPF', line=dict(color='blue')))
            fig_knee.update_layout(title="KNEE JOINT", xaxis_title="Time (s)", yaxis_title="Angle (Deg)", height=250, margin=dict(t=30, b=10))
            st.plotly_chart(fig_knee)

            # 3. ANKLE PLOT
            fig_ankle = go.Figure()
            fig_ankle.add_trace(go.Scatter(x=x_axis, y=df['Ankle'], name='Raw', line=dict(color='lightgray')))
            fig_ankle.add_trace(go.Scatter(x=x_axis, y=manual_lpf(df['Ankle'], alpha_val), name='LPF', line=dict(color='green')))
            fig_ankle.update_layout(title="ANKLE JOINT", xaxis_title="Time (s)", yaxis_title="Angle (Deg)", height=250, margin=dict(t=30, b=10))
            st.plotly_chart(fig_ankle)
            
    else:
        st.info("Upload sensor data to view the Gait Analysis.")

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