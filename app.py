import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ==========================================
# SIGNAL PROCESSING FUNCTIONS
# ==========================================

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

def process_gait_cycles(df, alpha, order, thresh_pct=50.0):
    """Segments data matching the lifespan of the Heel and Toe independently."""
    cycles = []
    
    df['Heel_LPF'] = manual_lpf(df['Heel'], alpha, order)
    df['Toe_LPF'] = manual_lpf(df['Toe'], alpha, order)
    
    h_max = df['Heel_LPF'].max()
    t_max = df['Toe_LPF'].max()
    
    h_thresh_val = h_max * (thresh_pct / 100.0)
    t_thresh_val = t_max * (thresh_pct / 100.0)
    
    ic_times = [] 
    ho_times = [] 
    ff_times = [] 
    to_times = [] 
    
    if h_max > 0 and t_max > 0:
        heel_bin = (df['Heel_LPF'] > h_thresh_val).astype(int)
        toe_bin = (df['Toe_LPF'] > t_thresh_val).astype(int)
        
        heel_edges = heel_bin.diff()
        toe_edges = toe_bin.diff()
        
        ic_times = df.loc[heel_edges == 1, 'Time'].tolist()   
        ho_times = df.loc[heel_edges == -1, 'Time'].tolist()  
        ff_times = df.loc[toe_edges == 1, 'Time'].tolist()    
        to_times = df.loc[toe_edges == -1, 'Time'].tolist()   
        
        ic_indices = df.index[heel_edges == 1].tolist()
        
        for i in range(len(ic_indices) - 1):
            start = ic_indices[i]
            end = ic_indices[i+1]
            
            c_df = df.iloc[start:end].copy()
            
            t_start = df['Time'].iloc[start]
            t_end = df['Time'].iloc[end]
            t_dur = t_end - t_start
            
            if t_dur <= 0: continue
            
            c_df['Gait_Pct'] = ((c_df['Time'] - t_start) / t_dur) * 100
            
            ff_candidates = df.index[(toe_edges == 1) & (df.index >= start) & (df.index < end)].tolist()
            ho_candidates = df.index[(heel_edges == -1) & (df.index > start) & (df.index < end)].tolist()
            to_candidates = df.index[(toe_edges == -1) & (df.index > start) & (df.index < end)].tolist()
            
            ff_pct = ((df['Time'].iloc[ff_candidates[0]] - t_start) / t_dur * 100) if ff_candidates else 15.0
            ho_pct = ((df['Time'].iloc[ho_candidates[-1]] - t_start) / t_dur * 100) if ho_candidates else 45.0
            to_pct = ((df['Time'].iloc[to_candidates[-1]] - t_start) / t_dur * 100) if to_candidates else 60.0
                
            cycles.append({
                'label': f'Siklus ke - {i+1}',
                'df': c_df,
                'duration': t_dur,
                'stance_pct': to_pct, 
                'swing_pct': 100 - to_pct,
                'cadence': 60 / t_dur,
                'ff_pct': ff_pct,
                'ho_pct': ho_pct,
                'to_pct': to_pct
            })
            
    return {
        'cycles': cycles, 'ic_times': ic_times, 'ff_times': ff_times, 
        'ho_times': ho_times, 'to_times': to_times, 
        'h_max': h_max, 't_max': t_max, 'h_thresh_val': h_thresh_val, 't_thresh_val': t_thresh_val
    }

# ==========================================
# STREAMLIT APPLICATION SETUP
# ==========================================

st.set_page_config(page_title="Wearable Sensor System", layout="wide")
st.title("Wearable Sensor System - Gait Analysis")

if 'sensor_data' not in st.session_state:
    st.session_state['sensor_data'] = None

# --- Sidebar ---
with st.sidebar:
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Sensor Data (TXT)", type=['txt'])
    
    if st.button("Clear Data"):
        st.session_state['sensor_data'] = None
        st.rerun()
        
    st.info("Filter settings are located in their respective analysis tabs for independent control.")

# --- Data Loading ---
if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), sep=r'\s+', header=None)
        df = df.dropna(axis=1, how='all')
        
        # Check if first row is headers, if so, skip it
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
            st.sidebar.error(f"Data format mismatch. Found {len(df.columns)} columns.")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")

data_exists = isinstance(st.session_state.get('sensor_data'), pd.DataFrame)

# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
if data_exists:
    df = st.session_state['sensor_data']
    dt = 0.01 # Assumed 100Hz sampling
    
    kinematic_tab, emg_tab = st.tabs(["Kinematic (Joint) Analysis", "EMG (Muscle) Analysis"])
    
    # ------------------------------------------
    # KINEMATIC TAB (3-STEP PROCESS)
    # ------------------------------------------
    with kinematic_tab:
        st.markdown("### Step 1: Raw Sensor Data")
        st.write("Initial, unfiltered data directly from the wearable sensors.")
        
        # --- STEP 1: RAW DATA PLOTS ---
        fig_raw = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=("RAW JOINT ANGLES", "RAW FSR (HEEL/TOE PRESSURE)")
        )
        
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Hip'], name='Hip', line=dict(color='red', width=1)), row=1, col=1)
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Knee'], name='Knee', line=dict(color='blue', width=1)), row=1, col=1)
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Ankle'], name='Ankle', line=dict(color='darkorange', width=1)), row=1, col=1)
        
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Heel'], name='Heel', line=dict(color='lightgreen', width=1)), row=2, col=1)
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Toe'], name='Toe', line=dict(color='thistle', width=1)), row=2, col=1)
        
        fig_raw.update_layout(height=450, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_raw, use_container_width=True)

        st.divider()

        # --- STEP 2: PREPROCESSING ---
        st.markdown("### Step 2: Filtering & Thresholding")
        st.write("Apply Low-Pass Filters (LPF) to smooth the FSR data and set thresholds to detect foot-ground contact.")
        
        k_col1, k_col2, k_col3 = st.columns(3)
        with k_col1:
            fsr_fc = st.slider("Cut-off Freq (Hz)", min_value=0.1, max_value=20.0, value=5.0, step=0.1, key='k_fc')
        with k_col2:
            fsr_order = st.slider("Filter Passes", min_value=1, max_value=10, value=2, step=1, key='k_order')
        with k_col3:
            gait_thresh = st.slider("Segmentation Threshold (%)", min_value=5.0, max_value=95.0, value=40.0, step=5.0, key='k_thresh')
            
        fsr_alpha = dt / ((1.0 / (2 * np.pi * fsr_fc)) + dt)
        
        # Process the cycles to get LPF data and thresholds
        cycle_data = process_gait_cycles(df, fsr_alpha, fsr_order, gait_thresh)
        h_thresh = cycle_data.get('h_thresh_val', 0)
        t_thresh = cycle_data.get('t_thresh_val', 0)

        fig_filter = go.Figure()
        fig_filter.add_trace(go.Scatter(x=df['Time'], y=df['Heel_LPF'], name='Heel LPF', line=dict(color='green', width=2)))
        fig_filter.add_trace(go.Scatter(x=df['Time'], y=df['Toe_LPF'], name='Toe LPF', line=dict(color='purple', width=2)))
        fig_filter.add_hline(y=h_thresh, line_dash="dot", line_color="green", annotation_text="Heel Thresh")
        fig_filter.add_hline(y=t_thresh, line_dash="dot", line_color="purple", annotation_text="Toe Thresh")
        
        fig_filter.update_layout(title="Filtered FSR Signals vs. Detection Thresholds", height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_filter, use_container_width=True)

        st.divider()

        # --- STEP 3: GAIT PHASE & PARAMETERS ---
        st.markdown("### Step 3: Gait Phase & Temporal Parameters")
        st.write("Final synchronized view of joint kinematics mapped against detected gait events.")
        
        cycles = cycle_data['cycles']
        cycle_opts = ["Full Record (Raw Time)"] + [c['label'] for c in cycles]
        selected_view = st.selectbox("Select Stride / View:", cycle_opts)
        
        # DYNAMIC METRICS: Averages for full record, Specifics for single cycle
        if cycles:
            c1, c2, c3, c4 = st.columns(4)
            if selected_view == "Full Record (Raw Time)":
                avg_stance = np.mean([c['stance_pct'] for c in cycles])
                std_stance = np.std([c['stance_pct'] for c in cycles])
                
                avg_swing = np.mean([c['swing_pct'] for c in cycles])
                std_swing = np.std([c['swing_pct'] for c in cycles])
                
                avg_dur = np.mean([c['duration'] for c in cycles])
                std_dur = np.std([c['duration'] for c in cycles])
                
                avg_cadence = np.mean([c['cadence'] for c in cycles])
                
                c1.metric("Avg Stance [%]", f"{avg_stance:.1f} ± {std_stance:.1f}")
                c2.metric("Avg Swing [%]", f"{avg_swing:.1f} ± {std_swing:.1f}")
                c3.metric("Avg Cycle Time [s]", f"{avg_dur:.2f} ± {std_dur:.2f}")
                c4.metric("Avg Cadence [strd/min]", f"{avg_cadence:.1f}")
            else:
                sel_cycle = next(c for c in cycles if c['label'] == selected_view)
                c1.metric("Stance Time [%]", f"{sel_cycle['stance_pct']:.1f}")
                c2.metric("Swing Time [%]", f"{sel_cycle['swing_pct']:.1f}")
                c3.metric("Cycle Time [s]", f"{sel_cycle['duration']:.2f}")
                c4.metric("Cadence [spm]", f"{sel_cycle['cadence']:.1f}")
            st.write("") # Spacer

        fig_final = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            subplot_titles=("NORMALIZED FSR + GAIT EVENTS", "JOINT ANGLES + GAIT EVENTS")
        )

        if selected_view == "Full Record (Raw Time)" or not cycles:
            # Full Time Series
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Heel_LPF'], name='Heel', line=dict(color='green')), row=1, col=1)
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Toe_LPF'], name='Toe', line=dict(color='purple')), row=1, col=1)
            
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Hip'], name='Hip', line=dict(color='red')), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Knee'], name='Knee', line=dict(color='blue')), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Ankle'], name='Ankle', line=dict(color='darkorange')), row=2, col=1)
            
            for ic in cycle_data['ic_times']: fig_final.add_vline(x=ic, line_color="green", opacity=0.4, row="all", col=1)
            for to in cycle_data['to_times']: fig_final.add_vline(x=to, line_color="purple", opacity=0.4, row="all", col=1)
            
            fig_final.update_xaxes(title_text="Time (s)", row=2, col=1)
                
        else:
            # Single Normalized Cycle
            c_df = sel_cycle['df']
            x_ax = c_df['Gait_Pct']
            
            h_norm = c_df['Heel_LPF'] / cycle_data['h_max'] if cycle_data['h_max'] > 0 else c_df['Heel_LPF']
            t_norm = c_df['Toe_LPF'] / cycle_data['t_max'] if cycle_data['t_max'] > 0 else c_df['Toe_LPF']

            fig_final.add_trace(go.Scatter(x=x_ax, y=h_norm, name='Heel (Norm)', line=dict(color='green')), row=1, col=1)
            fig_final.add_trace(go.Scatter(x=x_ax, y=t_norm, name='Toe (Norm)', line=dict(color='purple')), row=1, col=1)
            
            fig_final.add_trace(go.Scatter(x=x_ax, y=c_df['Hip'], name='Hip', line=dict(color='red')), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=x_ax, y=c_df['Knee'], name='Knee', line=dict(color='blue')), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=x_ax, y=c_df['Ankle'], name='Ankle', line=dict(color='darkorange')), row=2, col=1)
            
            fig_final.add_vline(x=0, line_color="green", line_dash="dash", row="all", col=1, annotation_text="Heel Strike")
            fig_final.add_vline(x=sel_cycle['to_pct'], line_color="purple", line_dash="dash", row="all", col=1, annotation_text="Toe Off")

            fig_final.update_xaxes(title_text="Normalized Gait Cycle (%)", row=2, col=1)

        fig_final.update_layout(height=600, margin=dict(t=40, b=40, l=10, r=10), showlegend=True)
        st.plotly_chart(fig_final, use_container_width=True)

    # ------------------------------------------
    # EMG TAB (ORIGINAL IMPLEMENTATION)
    # ------------------------------------------
    with emg_tab:
        st.subheader("1. EMG Filter Settings")
        
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            emg_fc = st.slider("EMG Cut-off (Hz)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
        with e_col2:
            emg_order = st.slider("EMG Filter Passes", min_value=1, max_value=10, value=1, step=1)
            
        emg_alpha = dt / ((1.0 / (2 * np.pi * emg_fc)) + dt)
        
        st.divider()
        
        emg_muscles = ['Gluteus_Maximus', 'Biceps_Femoris_Short', 'Biceps_Femoris_Long', 'Vastus_Medialis', 'Vastus_Lateralis', 'Rectus_Femoris', 'Soleus', 'Gastrocnemius', 'Tibialis_Anterior']
        muscle_thresholds = {}

        st.subheader("2. Individual Muscle Envelopes & Threshold Tuning")
        cols = st.columns(3)
        for index, muscle in enumerate(emg_muscles):
            col = cols[index % 3] 
            with col:
                display_name = muscle.replace('_', ' ').upper()
                st.markdown(f"**{display_name}**")
                thresh_pct = st.slider(f"Threshold (%)", min_value=1.0, max_value=50.0, value=5.0, step=0.5, key=f"thresh_{muscle}")
                muscle_thresholds[muscle] = thresh_pct
                
                rectified_data = manual_rectify(df[muscle])
                filtered_data = manual_lpf(rectified_data, emg_alpha, emg_order)
                max_val = max(filtered_data) if filtered_data else 0
                abs_threshold = (thresh_pct / 100.0) * max_val
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Time'], y=df[muscle], name='Raw', line=dict(color='lightgray', width=1)))
                fig.add_trace(go.Scatter(x=df['Time'], y=filtered_data, name='Envelope', line=dict(width=2)))
                fig.add_hline(y=abs_threshold, line_dash="dash", line_color="red")
                
                fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amp", height=200, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("3. Combined Muscle Activation Timing")
        
        fig_timing = go.Figure()
        emg_muscles_reversed = list(reversed(emg_muscles))
        
        for i, muscle in enumerate(emg_muscles_reversed):
            rectified_data = manual_rectify(df[muscle])
            envelope = manual_lpf(rectified_data, emg_alpha, emg_order)
            thresh_pct = muscle_thresholds[muscle]
            max_val = max(envelope) if envelope else 0
            abs_threshold = (thresh_pct / 100.0) * max_val
            
            active_state = [i if val >= abs_threshold else np.nan for val in envelope]
            
            fig_timing.add_trace(go.Scatter(x=df['Time'], y=active_state, mode='lines', name=muscle.replace('_', ' '), line=dict(width=15), hoverinfo='name+x'))

        fig_timing.update_layout(xaxis_title="Time (s)", yaxis=dict(tickmode='array', tickvals=list(range(len(emg_muscles_reversed))), ticktext=[m.replace('_', ' ').upper() for m in emg_muscles_reversed], showgrid=False, zeroline=False), height=450, margin=dict(t=30, b=10, l=150), showlegend=False)
        st.plotly_chart(fig_timing, use_container_width=True)
        
else:
    st.info("Please upload a Sensor Data file to begin Gait Analysis.")
