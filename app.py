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

def process_gait_cycles(df):
    if 'Heel' not in df.columns or 'Toe' not in df.columns:
        return []
        
    heel_diff = df['Heel'].diff()
    toe_diff = df['Toe'].diff()
    ic_indices = df.index[heel_diff == 1].tolist()

    cycles = []
    for i in range(len(ic_indices) - 1):
        start = ic_indices[i]
        end = ic_indices[i+1]
        cycle_df = df.loc[start:end-1].copy()

        time_start = df.loc[start, 'Time']
        t_cycle = df.loc[end, 'Time'] - time_start

        if t_cycle <= 0: continue

        cycle_df['GaitCycle'] = ((cycle_df['Time'] - time_start) / t_cycle) * 100

        to_idx_list = cycle_df.index[toe_diff.loc[start:end-1] == -1].tolist()
        ff_idx_list = cycle_df.index[toe_diff.loc[start:end-1] == 1].tolist()
        ho_idx_list = cycle_df.index[heel_diff.loc[start:end-1] == -1].tolist()

        to_idx = to_idx_list[0] if to_idx_list else None
        ff_idx = ff_idx_list[0] if ff_idx_list else None
        ho_idx = ho_idx_list[0] if ho_idx_list else None

        if to_idx is not None:
            t_to = df.loc[to_idx, 'Time'] - time_start
            p_to = (t_to / t_cycle) * 100
            p_ff = ((df.loc[ff_idx, 'Time'] - time_start) / t_cycle * 100) if ff_idx else 0
            p_ho = ((df.loc[ho_idx, 'Time'] - time_start) / t_cycle * 100) if ho_idx else 0

            cycles.append({
                'label': f"Siklus ke - {len(cycles) + 1}",
                'df': cycle_df,
                't_cycle': t_cycle,
                'ic_pct': 0.0,
                'ff_pct': p_ff,
                'ho_pct': p_ho,
                'to_pct': p_to,
                't_stance': p_to,
                't_swing': 100 - p_to,
                'cadence': 120 / t_cycle
            })
    return cycles

st.set_page_config(page_title="Wearable FES-Gait System", layout="wide")
st.title("Wearable FES-Gait System Graphical User Interface")

if 'sensor_data' not in st.session_state:
    st.session_state['sensor_data'] = None
if 'cycles' not in st.session_state:
    st.session_state['cycles'] = []

with st.sidebar:
    st.header("Data Management")
    uploaded_file = st.file_uploader("Upload Sensor Data (TXT)", type=['txt'])
    
    if st.button("Clear Data"):
        st.session_state['sensor_data'] = None
        st.session_state['cycles'] = []
        st.rerun()
        
    st.divider()
    st.header("Filter Settings")
    alpha_val = st.slider("LPF Alpha", min_value=0.01, max_value=1.0, value=0.15, step=0.01)

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
        st.session_state['cycles'] = process_gait_cycles(df)
    except Exception as e:
        pass

tab1, tab2 = st.tabs(["SENSOR SYSTEM (Gait Analysis)", "OPEN-LOOP FES SYSTEM"])

with tab1:
    if st.session_state['sensor_data'] is not None and st.session_state['cycles']:
        cycles = st.session_state['cycles']
        cycle_labels = [c['label'] for c in cycles]
        
        plot_col, param_col = st.columns([2, 1])
        
        with param_col:
            st.subheader("Parameters")
            selected_label = st.selectbox("CYCLE", cycle_labels)
            sel_cycle = next(c for c in cycles if c['label'] == selected_label)
            sel_df = sel_cycle['df']
            
            st.markdown("**Temporal Parameters**")
            col_t1, col_t2 = st.columns(2)
            col_t1.metric("IC [%time]", f"{sel_cycle['ic_pct']:.1f}")
            col_t1.metric("FF [%time]", f"{sel_cycle['ff_pct']:.1f}")
            col_t1.metric("HO [%time]", f"{sel_cycle['ho_pct']:.1f}")
            col_t2.metric("Tstance [%time]", f"{sel_cycle['t_stance']:.1f}")
            col_t2.metric("Tswing [%time]", f"{sel_cycle['t_swing']:.1f}")
            col_t2.metric("Tcycle [s]", f"{sel_cycle['t_cycle']:.2f}")
            col_t2.metric("Cad [strd/min]", f"{sel_cycle['cadence']:.1f}")
            
            st.divider()
            st.markdown("**Hip Joint Parameters**")
            st.metric("HIC [deg]", f"{sel_df['HipKal'].iloc[0]:.1f}" if 'HipKal' in sel_df else "0.0")
            
            st.divider()
            st.markdown("**Knee Joint Parameters**")
            st.metric("KIC [deg]", f"{sel_df['KneeKal'].iloc[0]:.1f}" if 'KneeKal' in sel_df else "0.0")

        with plot_col:
            st.subheader(f"Gait Analysis Plots ({selected_label})")
            
            x_axis = sel_df['GaitCycle']
            
            if 'HipKal' in sel_df.columns:
                fig_hip = go.Figure()
                fig_hip.add_trace(go.Scatter(x=x_axis, y=sel_df['HipKal'], name='Raw', line=dict(color='lightgray')))
                fig_hip.add_trace(go.Scatter(x=x_axis, y=manual_lpf(sel_df['HipKal'], alpha_val), name='LPF', line=dict(color='red')))
                fig_hip.update_layout(title="HIP JOINT", xaxis_title="gait cycle [%]", yaxis_title="Deg", height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_hip, use_container_width=True)

            if 'KneeKal' in sel_df.columns:
                fig_knee = go.Figure()
                fig_knee.add_trace(go.Scatter(x=x_axis, y=sel_df['KneeKal'], name='Raw', line=dict(color='lightgray')))
                fig_knee.add_trace(go.Scatter(x=x_axis, y=manual_lpf(sel_df['KneeKal'], alpha_val), name='LPF', line=dict(color='blue')))
                fig_knee.update_layout(title="KNEE JOINT", xaxis_title="gait cycle [%]", yaxis_title="Deg", height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_knee, use_container_width=True)

            if 'AnkleKal' in sel_df.columns:
                fig_ankle = go.Figure()
                fig_ankle.add_trace(go.Scatter(x=x_axis, y=sel_df['AnkleKal'], name='Raw', line=dict(color='lightgray')))
                fig_ankle.add_trace(go.Scatter(x=x_axis, y=manual_lpf(sel_df['AnkleKal'], alpha_val), name='LPF', line=dict(color='green')))
                fig_ankle.update_layout(title="ANKLE JOINT", xaxis_title="gait cycle [%]", yaxis_title="Deg", height=250, margin=dict(t=30, b=10))
                st.plotly_chart(fig_ankle, use_container_width=True)

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