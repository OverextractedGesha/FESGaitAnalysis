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
    
    # 1. Filter
    df['Heel_LPF'] = manual_lpf(df['Heel'], alpha, order)
    df['Toe_LPF'] = manual_lpf(df['Toe'], alpha, order)
    
    h_max = df['Heel_LPF'].max()
    t_max = df['Toe_LPF'].max()
    
    # 2. Normalize (0 to 1 based on peak)
    df['Heel_Norm'] = df['Heel_LPF'] / h_max if h_max > 0 else 0
    df['Toe_Norm'] = df['Toe_LPF'] / t_max if t_max > 0 else 0
    
    # 3. Threshold (now a simple decimal between 0 and 1)
    thresh_val = thresh_pct / 100.0
    
    ic_times = [] 
    ho_times = [] 
    ff_times = [] 
    to_times = [] 
    
    if h_max > 0 and t_max > 0:
        # Binarize based on NORMALIZED signal
        heel_bin = (df['Heel_Norm'] > thresh_val).astype(int)
        toe_bin = (df['Toe_Norm'] > thresh_val).astype(int)
        
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
                # Changed from 120 to 60 for accurate Strides/Min
                'cadence': 60 / t_dur, 
                'ff_pct': ff_pct,
                'ho_pct': ho_pct,
                'to_pct': to_pct
            })
            
    return {
        'cycles': cycles, 'ic_times': ic_times, 'ff_times': ff_times, 
        'ho_times': ho_times, 'to_times': to_times, 
        'thresh_val': thresh_val
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
    
    # ==========================================
    # GLOBAL GAIT SEGMENTATION (Shared by both tabs)
    # ==========================================
    st.sidebar.divider()
    st.sidebar.subheader("Global Step Detection (FSR)")
    fsr_fc = st.sidebar.slider("FSR Cut-off (Hz)", 0.1, 20.0, 5.0, 0.1)
    fsr_order = st.sidebar.slider("FSR Passes", 1, 10, 2, 1)
    gait_thresh = st.sidebar.slider("Step Threshold (%)", 5.0, 95.0, 40.0, 5.0)
    
    fsr_alpha = dt / ((1.0 / (2 * np.pi * fsr_fc)) + dt)
    cycle_data = process_gait_cycles(df, fsr_alpha, fsr_order, gait_thresh)
    cycles = cycle_data['cycles']

    kinematic_tab, emg_tab = st.tabs(["Kinematic (Joint) Analysis", "EMG (Muscle) Analysis"])
    
    # ------------------------------------------
    # KINEMATIC TAB
    # ------------------------------------------
    with kinematic_tab:
        st.markdown("### Step 1: Raw Sensor Data")
        fig_raw = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("RAW JOINT ANGLES", "RAW FSR"))
        
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Hip'], name='Hip', line=dict(color='red', width=1)), row=1, col=1)
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Knee'], name='Knee', line=dict(color='blue', width=1)), row=1, col=1)
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Ankle'], name='Ankle', line=dict(color='darkorange', width=1)), row=1, col=1)
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Heel'], name='Heel', line=dict(color='lightgreen', width=1)), row=2, col=1)
        fig_raw.add_trace(go.Scatter(x=df['Time'], y=df['Toe'], name='Toe', line=dict(color='thistle', width=1)), row=2, col=1)
        
        fig_raw.update_layout(height=450, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_raw, use_container_width=True)

        st.divider()

        st.markdown("### Step 2: FSR Preprocessing & Edge Detection")
        st.write("FSR data is filtered and **normalized (0 to 1)** based on peak pressure before applying the threshold.")
        fig_filter = go.Figure()
        fig_filter.add_trace(go.Scatter(x=df['Time'], y=df['Heel_Norm'], name='Heel (Norm)', line=dict(color='green', width=2)))
        fig_filter.add_trace(go.Scatter(x=df['Time'], y=df['Toe_Norm'], name='Toe (Norm)', line=dict(color='purple', width=2)))
        
        norm_thresh = cycle_data.get('thresh_val', 0)
        fig_filter.add_hline(y=norm_thresh, line_dash="dot", line_color="black", annotation_text=f"Global Threshold ({gait_thresh}%)")
        
        fig_filter.update_layout(height=300, yaxis_title="Normalized Amp", margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_filter, use_container_width=True)

        st.divider()

        st.markdown("### Step 3: Kinematic Gait Phases")
        cycle_opts = ["Full Record (Raw Time)"] + [c['label'] for c in cycles]
        selected_view = st.selectbox("Select Stride / View:", cycle_opts, key="kinematic_view")
        
        if cycles:
            c1, c2, c3, c4 = st.columns(4)
            if selected_view == "Full Record (Raw Time)":
                c1.metric("Avg Stance [%]", f"{np.mean([c['stance_pct'] for c in cycles]):.1f}")
                c2.metric("Avg Swing [%]", f"{np.mean([c['swing_pct'] for c in cycles]):.1f}")
                c3.metric("Avg Cycle Time [s]", f"{np.mean([c['duration'] for c in cycles]):.2f}")
                c4.metric("Avg Cadence [strd/min]", f"{np.mean([c['cadence'] for c in cycles]):.1f}")
            else:
                sel_cycle = next(c for c in cycles if c['label'] == selected_view)
                c1.metric("Stance Time [%]", f"{sel_cycle['stance_pct']:.1f}")
                c2.metric("Swing Time [%]", f"{sel_cycle['swing_pct']:.1f}")
                c3.metric("Cycle Time [s]", f"{sel_cycle['duration']:.2f}")
                c4.metric("Cadence [strd/min]", f"{sel_cycle['cadence']:.1f}")

        fig_final = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("NORMALIZED FSR", "JOINT ANGLES"))

        if selected_view == "Full Record (Raw Time)" or not cycles:
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Heel_Norm'], name='Heel', line=dict(color='green')), row=1, col=1)
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Toe_Norm'], name='Toe', line=dict(color='purple')), row=1, col=1)
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Hip'], name='Hip', line=dict(color='red')), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Knee'], name='Knee', line=dict(color='blue')), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=df['Time'], y=df['Ankle'], name='Ankle', line=dict(color='darkorange')), row=2, col=1)
            for ic in cycle_data['ic_times']: fig_final.add_vline(x=ic, line_color="green", opacity=0.4, row="all", col=1)
            for to in cycle_data['to_times']: fig_final.add_vline(x=to, line_color="purple", opacity=0.4, row="all", col=1)
        else:
            c_df = sel_cycle['df']
            fig_final.add_trace(go.Scatter(x=c_df['Gait_Pct'], y=c_df['Heel_Norm'], name='Heel', line=dict(color='green')), row=1, col=1)
            fig_final.add_trace(go.Scatter(x=c_df['Gait_Pct'], y=c_df['Toe_Norm'], name='Toe', line=dict(color='purple')), row=1, col=1)
            fig_final.add_trace(go.Scatter(x=c_df['Gait_Pct'], y=c_df['Hip'], name='Hip', line=dict(color='red')), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=c_df['Gait_Pct'], y=c_df['Knee'], name='Knee', line=dict(color='blue')), row=2, col=1)
            fig_final.add_trace(go.Scatter(x=c_df['Gait_Pct'], y=c_df['Ankle'], name='Ankle', line=dict(color='darkorange')), row=2, col=1)
            fig_final.add_vline(x=0, line_color="green", line_dash="dash", row="all", col=1, annotation_text="Heel Strike")
            fig_final.add_vline(x=sel_cycle['to_pct'], line_color="purple", line_dash="dash", row="all", col=1, annotation_text="Toe Off")

        fig_final.update_layout(height=600, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_final, use_container_width=True)

    # ------------------------------------------
    # EMG TAB 
    # ------------------------------------------
    with emg_tab:
        emg_muscles = ['Gluteus_Maximus', 'Biceps_Femoris_Short', 'Biceps_Femoris_Long', 'Vastus_Medialis', 'Vastus_Lateralis', 'Rectus_Femoris', 'Soleus', 'Gastrocnemius', 'Tibialis_Anterior']
        
        st.markdown("### Step 1: Raw EMG Data")
        st.write("Displaying a subset of raw muscle signals for quality check.")
        fig_emg_raw = go.Figure()
        # Just plot the first 3 muscles to avoid clutter, user can check standard data quality here
        for m in emg_muscles[:3]: 
            fig_emg_raw.add_trace(go.Scatter(x=df['Time'], y=df[m], name=m.replace('_', ' ')))
        fig_emg_raw.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_emg_raw, use_container_width=True)
        
        st.divider()

        st.markdown("### Step 2: Linear Envelopes & Normalization")
        st.write("Muscles are rectified, filtered, and **normalized (0 to 1)** to their peak activation. Thresholds are evaluated against this normalized envelope.")
        
        e_col1, e_col2 = st.columns(2)
        with e_col1: emg_fc = st.slider("EMG Cut-off (Hz)", 0.1, 20.0, 3.0, 0.1)
        with e_col2: emg_order = st.slider("EMG Filter Passes", 1, 10, 1, 1)
        emg_alpha = dt / ((1.0 / (2 * np.pi * emg_fc)) + dt)
        
        muscle_envelopes_norm = {}
        muscle_thresholds_vals = {}
        
        # Removed 3x3 columns - now a vertical scroll layout
        for muscle in emg_muscles:
            st.markdown(f"#### {muscle.replace('_', ' ').upper()}")
            
            # Use columns just to put the slider next to the graph title if desired, 
            # but keeping it simple here:
            thresh_pct = st.slider(f"Activation Threshold (%)", 1.0, 50.0, 5.0, 0.5, key=f"t_{muscle}")
            
            # Process FULL record
            rectified_data = manual_rectify(df[muscle])
            envelope = manual_lpf(rectified_data, emg_alpha, emg_order)
            
            # Normalize (0 to 1)
            max_val = max(envelope) if envelope else 0
            env_norm = [v / max_val for v in envelope] if max_val > 0 else envelope
            muscle_envelopes_norm[muscle] = env_norm
            
            # Threshold is now simply the percentage (e.g., 5% = 0.05)
            norm_threshold = thresh_pct / 100.0
            muscle_thresholds_vals[muscle] = norm_threshold
            
            # Plot preview (Full Record, Full Width)
            fig_env = go.Figure()
            # We scale the raw data visually just so it fits on the same 0-1 axis for comparison
            raw_scaled = [v / max_val for v in df[muscle]] if max_val > 0 else df[muscle]
            
            fig_env.add_trace(go.Scatter(x=df['Time'], y=raw_scaled, name='Raw (Scaled)', line=dict(color='lightgray', width=1)))
            fig_env.add_trace(go.Scatter(x=df['Time'], y=env_norm, name='Normalized Env', line=dict(width=2, color='royalblue')))
            fig_env.add_hline(y=norm_threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({thresh_pct}%)")
            
            fig_env.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), showlegend=True, yaxis_title="Norm Amp")
            st.plotly_chart(fig_env, use_container_width=True)
            st.write("") # Small spacer

        st.divider()

        st.markdown("### Step 3: Muscle Activation Timing")
        st.write("View continuous activation, or slice the processed data to view a single stride.")
        
        emg_cycle_opts = ["Full Record (Raw Time)"] + [c['label'] for c in cycles]
        emg_view = st.selectbox("Select Stride / View:", emg_cycle_opts, key="emg_view")
        
        fig_timing = go.Figure()
        emg_muscles_reversed = list(reversed(emg_muscles))
        
        for i, muscle in enumerate(emg_muscles_reversed):
            env_norm = muscle_envelopes_norm[muscle]
            norm_threshold = muscle_thresholds_vals[muscle]
            
            # Determine active state for FULL record based on normalized data
            active_state = [i if val >= norm_threshold else np.nan for val in env_norm]
            
            if emg_view == "Full Record (Raw Time)" or not cycles:
                # Plot full record
                fig_timing.add_trace(go.Scatter(x=df['Time'], y=active_state, mode='lines', name=muscle, line=dict(width=15)))
            else:
                # Slicing the PRE-PROCESSED data to match the specific cycle
                sel_cycle = next(c for c in cycles if c['label'] == emg_view)
                start_idx = sel_cycle['df'].index[0]
                end_idx = sel_cycle['df'].index[-1] + 1
                
                cycle_active = active_state[start_idx:end_idx]
                cycle_pct = sel_cycle['df']['Gait_Pct']
                
                fig_timing.add_trace(go.Scatter(x=cycle_pct, y=cycle_active, mode='lines', name=muscle, line=dict(width=15)))
        
        # Format the X-axis depending on view
        x_title = "Time (s)" if emg_view == "Full Record (Raw Time)" else "Normalized Gait Cycle (%)"
        
        fig_timing.update_layout(
            xaxis_title=x_title, 
            yaxis=dict(tickmode='array', tickvals=list(range(len(emg_muscles_reversed))), ticktext=[m.replace('_', ' ').upper() for m in emg_muscles_reversed], showgrid=False, zeroline=False), 
            height=450, margin=dict(t=30, b=10, l=150), showlegend=False
        )
        st.plotly_chart(fig_timing, use_container_width=True)
        
else:
    st.info("Please upload a Sensor Data file to begin Gait Analysis.")
