import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- Custom Signal Processing Functions ---

def manual_rectify(data_series):
    data = pd.to_numeric(data_series, errors='coerce').fillna(0)
    return data.apply(lambda x: x if x >= 0 else -x)

def manual_lpf(data_series, alpha, order):
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

def process_gait_cycles(df):
    """Segments data and explicitly saves the timestamps for visualization."""
    cycles = []
    ic_times = []
    to_times = []
    
    h_max = df['Heel'].max()
    t_max = df['Toe'].max()
    
    if h_max > 0:
        heel_bin = (df['Heel'] > h_max * 0.5).astype(int)
        toe_bin = (df['Toe'] > t_max * 0.5).astype(int)
        
        heel_edges = heel_bin.diff()
        ic_indices = df.index[heel_edges == 1].tolist()
        toe_edges = toe_bin.diff()
        
        for i in range(len(ic_indices) - 1):
            start = ic_indices[i]
            end = ic_indices[i+1]
            
            c_df = df.iloc[start:end].copy()
            
            t_start = df['Time'].iloc[start]
            t_end = df['Time'].iloc[end]
            t_dur = t_end - t_start
            
            if t_dur <= 0: continue
            
            ic_times.append(t_start)
            c_df['Gait_Pct'] = ((c_df['Time'] - t_start) / t_dur) * 100
            
            to_candidates = df.index[(toe_edges == -1) & (df.index > start) & (df.index < end)].tolist()
            to_pct = 60.0 
            if to_candidates:
                to_idx = to_candidates[0]
                to_time = df['Time'].iloc[to_idx]
                to_times.append(to_time)
                to_pct = ((to_time - t_start) / t_dur) * 100
            else:
                to_times.append(t_start + t_dur * 0.6)
                
            cycles.append({
                'label': f'Siklus ke - {i+1}',
                'df': c_df,
                'duration': t_dur,
                'stance_pct': to_pct,
                'swing_pct': 100 - to_pct,
                'cadence': 120 / t_dur 
            })
            
    return {'cycles': cycles, 'ic_times': ic_times, 'to_times': to_times, 'h_max': h_max, 't_max': t_max}

# --- Streamlit Application ---

st.set_page_config(page_title="Wearable FES-Gait System", layout="wide")
st.title("Wearable FES-Gait System Dashboard")

if 'sensor_data' not in st.session_state:
    st.session_state['sensor_data'] = None
if 'cycle_data' not in st.session_state:
    st.session_state['cycle_data'] = {'cycles': [], 'ic_times': [], 'to_times': []}

with st.sidebar:
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Sensor Data (TXT)", type=['txt'])
    
    if st.button("Clear Data"):
        st.session_state['sensor_data'] = None
        st.session_state['cycle_data'] = {'cycles': [], 'ic_times': [], 'to_times': []}
        st.rerun()
        
    st.divider()
    st.header("Filter Settings")
    fc = st.slider("Cut-off Frequency (Hz)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    filter_order = st.slider("Filter Order (Passes)", min_value=1, max_value=5, value=1, step=1)
    
    dt = 0.01 
    rc = 1.0 / (2 * np.pi * fc)
    alpha_val = dt / (rc + dt)

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
            st.session_state['cycle_data'] = process_gait_cycles(df)
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        else:
            st.sidebar.error(f"Data format mismatch. Found {len(df.columns)} columns.")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

tab1, tab2 = st.tabs(["SENSOR SYSTEM (Gait Analysis)", "OPEN-LOOP FES SYSTEM"])
data_exists = isinstance(st.session_state.get('sensor_data'), pd.DataFrame)

with tab1:
    if data_exists:
        df = st.session_state['sensor_data']
        cycle_data = st.session_state['cycle_data']
        cycles = cycle_data['cycles']
        
        kinematic_tab, emg_tab = st.tabs(["Kinematic (Joint) Analysis", "EMG (Muscle) Analysis"])
        
        # ==========================================
        # KINEMATIC MENU (Visualizing the Process)
        # ==========================================
        with kinematic_tab:
            st.subheader("Process 1: Thresholding & Cycle Segmentation")
            st.markdown("Visualizing Initial Contact (IC) and Toe Off (TO) detection from raw foot switches.")
            
            # --- 1. SEGMENTATION VISUALIZATION ---
            fig_seg = go.Figure()
            fig_seg.add_trace(go.Scatter(x=df['Time'], y=df['Heel'], name='Heel Sensor', line=dict(color='blue')))
            fig_seg.add_trace(go.Scatter(x=df['Time'], y=df['Toe'], name='Toe Sensor', line=dict(color='orange')))
            
            # Plot dynamic threshold lines
            h_thresh = cycle_data.get('h_max', 0) * 0.5
            fig_seg.add_hline(y=h_thresh, line_dash="dot", line_color="gray", annotation_text="Detection Threshold")
            
            # Draw vertical markers for IC and TO
            for ic in cycle_data['ic_times']:
                fig_seg.add_vline(x=ic, line_color="green", line_width=2, annotation_text="IC (Start)")
            for to in cycle_data['to_times']:
                fig_seg.add_vline(x=to, line_color="red", line_dash="dash", line_width=1, annotation_text="TO")
                
            fig_seg.update_layout(xaxis_title="Raw Time (s)", yaxis_title="Sensor Value", height=250, margin=dict(t=30, b=10))
            st.plotly_chart(fig_seg, use_container_width=True)

            st.divider()

            # --- 2. TIME NORMALIZATION VISUALIZATION ---
            st.subheader("Process 2: Time Normalization (0-100% Gait Cycle)")
            
            plot_col, param_col = st.columns([2, 1])
            
            with param_col:
                cycle_opts = ["Full Record (Raw Time)"] + [c['label'] for c in cycles]
                selected_view = st.selectbox("Select View:", cycle_opts)
                
                st.markdown("**Temporal Parameters**")
                if selected_view == "Full Record (Raw Time)" or not cycles:
                    st.info("Select a segmented cycle to view normalized metrics.")
                    st.metric("Total Record Time", f"{df['Time'].max():.2f} s")
                    st.metric("Cycles Detected", len(cycles))
                else:
                    sel_cycle = next(c for c in cycles if c['label'] == selected_view)
                    col1, col2 = st.columns(2)
                    col1.metric("T-Stance [%]", f"{sel_cycle['stance_pct']:.1f}")
                    col1.metric("T-Swing [%]", f"{sel_cycle['swing_pct']:.1f}")
                    col2.metric("Cycle Time [s]", f"{sel_cycle['duration']:.2f}")
                    col2.metric("Cadence [spm]", f"{sel_cycle['cadence']:.1f}")
                    
                    st.divider()
                    st.markdown("**Peak Kinematics (Filtered)**")
                    f_hip = manual_lpf(sel_cycle['df']['Hip'], alpha_val, filter_order)
                    f_knee = manual_lpf(sel_cycle['df']['Knee'], alpha_val, filter_order)
                    f_ankle = manual_lpf(sel_cycle['df']['Ankle'], alpha_val, filter_order)
                    
                    st.metric("Max Hip Flexion [deg]", f"{max(f_hip):.1f}")
                    st.metric("Max Knee Flexion [deg]", f"{max(f_knee):.1f}")
                    st.metric("Max Ankle Dorsi [deg]", f"{max(f_ankle):.1f}")

            with plot_col:
                fig_joints = go.Figure()
                
                if selected_view == "Full Record (Raw Time)" or not cycles:
                    x_axis = df['Time']
                    x_title = "Time (s)"
                    plot_hip = manual_lpf(df['Hip'], alpha_val, filter_order)
                    plot_knee = manual_lpf(df['Knee'], alpha_val, filter_order)
                    plot_ankle = manual_lpf(df['Ankle'], alpha_val, filter_order)
                    
                    fig_joints.add_trace(go.Scatter(x=x_axis, y=plot_hip, name='Hip', line=dict(color='red')))
                    fig_joints.add_trace(go.Scatter(x=x_axis, y=plot_knee, name='Knee', line=dict(color='blue')))
                    fig_joints.add_trace(go.Scatter(x=x_axis, y=plot_ankle, name='Ankle', line=dict(color='green')))
                else:
                    sel_cycle = next(c for c in cycles if c['label'] == selected_view)
                    c_df = sel_cycle['df']
                    x_axis = c_df['Gait_Pct']
                    x_title = "Normalized Gait Cycle (%)"
                    plot_hip = manual_lpf(c_df['Hip'], alpha_val, filter_order)
                    plot_knee = manual_lpf(c_df['Knee'], alpha_val, filter_order)
                    plot_ankle = manual_lpf(c_df['Ankle'], alpha_val, filter_order)
                    
                    fig_joints.add_trace(go.Scatter(x=x_axis, y=plot_hip, name='Hip', line=dict(color='red', width=3)))
                    fig_joints.add_trace(go.Scatter(x=x_axis, y=plot_knee, name='Knee', line=dict(color='blue', width=3)))
                    fig_joints.add_trace(go.Scatter(x=x_axis, y=plot_ankle, name='Ankle', line=dict(color='green', width=3)))
                    
                    # Vertical line splitting Stance vs Swing phase
                    fig_joints.add_vline(x=sel_cycle['stance_pct'], line_dash="dash", line_color="gray", annotation_text="Toe Off (Stance → Swing)")

                fig_joints.update_layout(xaxis_title=x_title, yaxis_title="Angle (Deg)", height=400, margin=dict(t=30, b=10))
                st.plotly_chart(fig_joints, use_container_width=True)

        # ==========================================
        # EMG MENU (Remains unchanged)
        # ==========================================
        with emg_tab:
            emg_muscles = ['Gluteus_Maximus', 'Biceps_Femoris_Short', 'Biceps_Femoris_Long', 'Vastus_Medialis', 'Vastus_Lateralis', 'Rectus_Femoris', 'Soleus', 'Gastrocnemius', 'Tibialis_Anterior']
            muscle_thresholds = {}

            st.subheader("Individual Muscle Envelopes & Threshold Tuning")
            cols = st.columns(3)
            for index, muscle in enumerate(emg_muscles):
                col = cols[index % 3] 
                with col:
                    display_name = muscle.replace('_', ' ').upper()
                    st.markdown(f"**{display_name}**")
                    thresh_pct = st.slider(f"Threshold (%)", min_value=1.0, max_value=50.0, value=5.0, step=0.5, key=f"thresh_{muscle}")
                    muscle_thresholds[muscle] = thresh_pct
                    
                    rectified_data = manual_rectify(df[muscle])
                    filtered_data = manual_lpf(rectified_data, alpha_val, filter_order)
                    max_val = max(filtered_data) if filtered_data else 0
                    abs_threshold = (thresh_pct / 100.0) * max_val
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['Time'], y=df[muscle], name='Raw', line=dict(color='lightgray', width=1)))
                    fig.add_trace(go.Scatter(x=df['Time'], y=filtered_data, name='Envelope', line=dict(width=2)))
                    fig.add_hline(y=abs_threshold, line_dash="dash", line_color="red")
                    
                    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amp", height=200, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Combined Muscle Activation Timing")
            
            fig_timing = go.Figure()
            emg_muscles_reversed = list(reversed(emg_muscles))
            
            for i, muscle in enumerate(emg_muscles_reversed):
                rectified_data = manual_rectify(df[muscle])
                envelope = manual_lpf(rectified_data, alpha_val, filter_order)
                thresh_pct = muscle_thresholds[muscle]
                max_val = max(envelope) if envelope else 0
                abs_threshold = (thresh_pct / 100.0) * max_val
                
                active_state = [i if val >= abs_threshold else np.nan for val in envelope]
                
                fig_timing.add_trace(go.Scatter(x=df['Time'], y=active_state, mode='lines', name=muscle.replace('_', ' '), line=dict(width=15), hoverinfo='name+x'))

            fig_timing.update_layout(xaxis_title="Time (s)", yaxis=dict(tickmode='array', tickvals=list(range(len(emg_muscles_reversed))), ticktext=[m.replace('_', ' ').upper() for m in emg_muscles_reversed], showgrid=False, zeroline=False), height=450, margin=dict(t=30, b=10, l=150), showlegend=False)
            st.plotly_chart(fig_timing, use_container_width=True)
            
    else:
        st.info("Upload sensor data to view the Gait Analysis.")

with tab2:
    st.subheader("Open-Loop FES Configuration")
    # ... [Tab 2 Open Loop FES content remains unchanged]
    fes_control_col, fes_plot_col = st.columns([1, 3])
    with fes_control_col:
        st.button("START FES")
        st.button("STOP FES")
    with fes_plot_col:
        fig_boost = go.Figure()
        fig_boost.update_layout(title="Boost Voltage", height=200)
        st.plotly_chart(fig_boost, use_container_width=True)