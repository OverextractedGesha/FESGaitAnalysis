import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- Custom Signal Processing Functions ---

def manual_rectify(data_series):
    """Manually rectifies the signal by converting negative values to positive (Absolute Value)"""
    data = pd.to_numeric(data_series, errors='coerce').fillna(0)
    return data.apply(lambda x: x if x >= 0 else -x)

def manual_lpf(data_series, alpha, order):
    """Applies a manual IIR Low Pass Filter (Cascaded for higher orders)"""
    data = pd.to_numeric(data_series, errors='coerce').fillna(0).tolist()
    if not data: return []
    
    current_data = data
    for _ in range(order):
        filtered = [current_data[0]] 
        for i in range(1, len(current_data)):
            val = alpha * current_data[i] + (1 - alpha) * filtered[-1]
            filtered.append(val)
        current_data = filtered
        
    return current_data

# --- Streamlit Application ---

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
    st.header("EMG Filter Settings")
    
    fc = st.slider("Cut-off Frequency (Hz)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    filter_order = st.slider("Filter Order (Passes)", min_value=1, max_value=5, value=1, step=1)
    
    # 100Hz assumed sampling rate
    dt = 0.01 
    rc = 1.0 / (2 * np.pi * fc)
    alpha_val = dt / (rc + dt)
    
    st.caption(f"Calculated internal \u03B1: {alpha_val:.4f}")
    
    st.divider()
    st.header("Activation Threshold")
    threshold_pct = st.slider("Threshold (% of Max)", min_value=1.0, max_value=30.0, value=5.0, step=0.5,
                              help="Signal values above this percentage of the muscle's maximum are plotted as 'Active'.")

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), sep=r'\s+', header=None)
        df = df.dropna(axis=1, how='all')
        
        if isinstance(df.iloc[0, 0], str) and not df.iloc[0, 0].replace('.','',1).isdigit():
            df = df.iloc[1:].reset_index(drop=True)
            
        if len(df.columns) >= 15:
            df = df.iloc[:, :15] 
            df.columns = [
                'Time', 'Heel', 'Toe', 'Hip', 'Knee', 'Ankle', 
                'Gluteus_Maximus', 'Biceps_Femoris_Short', 'Biceps_Femoris_Long', 
                'Vastus_Medialis', 'Vastus_Lateralis', 'Rectus_Femoris', 
                'Soleus', 'Gastrocnemius', 'Tibialis_Anterior'
            ]
            
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
        x_axis = df['Time']
        
        kinematic_tab, emg_tab = st.tabs(["Kinematic (Joint) Analysis", "EMG (Muscle) Analysis"])
        
        # ==========================================
        # KINEMATIC MENU
        # ==========================================
        with kinematic_tab:
            plot_col, param_col = st.columns([2, 1])
            
            with param_col:
                st.subheader("Joint Parameters")
                st.metric("Max Hip Angle [deg]", f"{df['Hip'].max():.1f}")
                st.metric("Max Knee Angle [deg]", f"{df['Knee'].max():.1f}")
                st.metric("Max Ankle Angle [deg]", f"{df['Ankle'].max():.1f}")

            with plot_col:
                st.subheader("Kinematic Joint Angles")
                fig_joints = go.Figure()
                fig_joints.add_trace(go.Scatter(x=x_axis, y=df['Hip'], name='Hip', line=dict(color='red')))
                fig_joints.add_trace(go.Scatter(x=x_axis, y=df['Knee'], name='Knee', line=dict(color='blue')))
                fig_joints.add_trace(go.Scatter(x=x_axis, y=df['Ankle'], name='Ankle', line=dict(color='green')))
                fig_joints.update_layout(title="LOWER LIMB JOINTS (Raw)", xaxis_title="Time (s)", yaxis_title="Angle (Deg)", height=350, margin=dict(t=30, b=10))
                st.plotly_chart(fig_joints, use_container_width=True)

        # ==========================================
        # EMG MENU
        # ==========================================
        with emg_tab:
            emg_muscles = [
                'Gluteus_Maximus', 'Biceps_Femoris_Short', 'Biceps_Femoris_Long',
                'Vastus_Medialis', 'Vastus_Lateralis', 'Rectus_Femoris',
                'Soleus', 'Gastrocnemius', 'Tibialis_Anterior'
            ]

            # --- 1. MUSCLE ACTIVATION TIMING CHART (ON/OFF) ---
            st.subheader("Muscle Activation Timing")
            st.markdown(f"*(Blocks indicate muscle envelope exceeds {threshold_pct}% of its maximum)*")
            
            fig_timing = go.Figure()
            # Reverse the list so the first muscle appears at the top of the y-axis
            emg_muscles_reversed = list(reversed(emg_muscles))
            
            for i, muscle in enumerate(emg_muscles_reversed):
                rectified_data = manual_rectify(df[muscle])
                envelope = manual_lpf(rectified_data, alpha_val, filter_order)
                
                # Calculate the activation threshold
                max_val = max(envelope) if envelope else 0
                threshold = (threshold_pct / 100.0) * max_val
                
                # Map to Y-index if active, else np.nan to break the line
                active_state = [i if val >= threshold else np.nan for val in envelope]
                
                fig_timing.add_trace(go.Scatter(
                    x=x_axis, 
                    y=active_state, 
                    mode='lines', 
                    name=muscle.replace('_', ' '),
                    line=dict(width=15), # Thick line mimics a Gantt bar
                    hoverinfo='name+x'
                ))

            fig_timing.update_layout(
                xaxis_title="Time (s)",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(emg_muscles_reversed))),
                    ticktext=[m.replace('_', ' ').upper() for m in emg_muscles_reversed],
                    showgrid=False,
                    zeroline=False
                ),
                height=450,
                margin=dict(t=30, b=10, l=150),
                showlegend=False
            )
            st.plotly_chart(fig_timing, use_container_width=True)
            
            st.divider()

            # --- 2. DETAILED RAW VS ENVELOPE CHARTS ---
            st.subheader("Individual Muscle Envelopes")
            
            # Use columns to lay them out efficiently (3 columns wide)
            cols = st.columns(3)
            
            for index, muscle in enumerate(emg_muscles):
                col = cols[index % 3] # Distribute across the 3 columns
                with col:
                    rectified_data = manual_rectify(df[muscle])
                    filtered_data = manual_lpf(rectified_data, alpha_val, filter_order)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_axis, y=df[muscle], name='Raw', line=dict(color='lightgray', width=1)))
                    fig.add_trace(go.Scatter(x=x_axis, y=filtered_data, name='Envelope', line=dict(width=2)))
                    
                    display_name = muscle.replace('_', ' ').upper()
                    fig.update_layout(
                        title=dict(text=display_name, font=dict(size=12)), 
                        xaxis_title="Time (s)", 
                        yaxis_title="Amp", 
                        height=200, 
                        margin=dict(t=30, b=10, l=10, r=10),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
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
        st.plotly_chart(fig_boost, use_container_width=True)
        fig_fes_hip = go.Figure()
        fig_fes_hip.update_layout(title="HIP JOINT (FES Response)", height=200, margin=dict(t=30, b=10))
        st.plotly_chart(fig_fes_hip, use_container_width=True)