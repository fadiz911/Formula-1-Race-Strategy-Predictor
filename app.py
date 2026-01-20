import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import fastf1
import logging
import random
import os

# Suppress logs
logging.getLogger('fastf1').setLevel(logging.ERROR)

# Ensure cache is enabled globally
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

from src.predictor import load_prediction_data, filter_practice_long_runs, get_driver_recent_form
from src.telemetry import get_driver_laps
from src.strategy_engine import train_tire_model, calculate_field_deltas
from src.optimizer import optimize_strategy
from src.simulation import simulate_race_history
from src.evaluator import evaluate_sim_performance
from src.feature_engineering import (
    extract_driver_historical_features,
    extract_track_characteristics,
    extract_driver_compound_affinity
) 

# --- CONFIG ---
st.set_page_config(page_title="F1 Strategy Command", page_icon="🏎️", layout="wide")
st.markdown("""<style>.stApp { background-color: #0e1117; color: white; }</style>""", unsafe_allow_html=True)

# --- COLORS ---
TEAM_COLORS = {
    'VER': '#0600ef', 'PER': '#0600ef', 'HAM': '#00d2be', 'RUS': '#00d2be',
    'LEC': '#dc0000', 'SAI': '#dc0000', 'NOR': '#ff8700', 'PIA': '#ff8700',
    'ALO': '#006f62', 'STR': '#006f62', 'GAS': '#ff00a0', 'OCO': '#ff00a0',
    'ALB': '#005aff', 'COL': '#005aff', 'TSU': '#6692ff', 'LAW': '#6692ff'
}
def get_driver_color(d): return TEAM_COLORS.get(d, '#ffffff')

# --- HELPERS ---
class AnchoredModel:
    def __init__(self, quali_pace, deg_slope, compound_offset):
        self.base_pace = float(quali_pace) + float(compound_offset)
        self.deg_slope = float(deg_slope)
    def predict(self, X):
        X = np.array(X)
        if len(X.shape) > 1: lap_age = X[:, 0]
        else: lap_age = X.flatten() 
        prediction = self.base_pace + (lap_age * self.deg_slope)
        return float(prediction.item()) if prediction.size == 1 else prediction.astype(float)

def format_strategy(strat_name, strat_laps, total_laps):
    if "1-Stop" in strat_name:
        return f"S {strat_laps} / H {total_laps - strat_laps}"
    elif "2-Stop" in strat_name:
        l1, l2_end = strat_laps
        return f"S {l1} / H {l2_end - l1} / S {total_laps - l2_end}"
    return "N/A"

def get_round_from_name(year, race_name):
    map_modern = {"Bahrain":1, "Saudi":2, "Australia":3, "Japan":4, "China":5, "Miami":6, "Emilia":7, "Monaco":8, "Canada":9, "Spain":10, "Austria":11, "Britain":12, "Hungary":13, "Belgium":14, "Netherlands":15, "Italy":16, "Azerbaijan":17, "Singapore":18, "United":19, "Mexico":20, "Brazil":21, "Las":22, "Qatar":23, "Abu":24}
    for k, v in map_modern.items():
        if k in race_name: return v
    return None

def train_anchored_models(session, driver, field_data, fastest_laps):
    try: quali_pace = fastest_laps[driver]
    except: quali_pace = fastest_laps.median()
    deg_slopes = {'SOFT': 0.12, 'MEDIUM': 0.08, 'HARD': 0.05}
    try:
        laps = filter_practice_long_runs(get_driver_laps(session, driver))
        if not laps.empty:
            for c in ['SOFT', 'MEDIUM', 'HARD']:
                model = train_tire_model(laps, c)
                if model:
                    t1, t2 = model.predict([[1]]), model.predict([[2]])
                    deg_slopes[c] = max(0.0, t2 - t1)
    except: pass
    models = {}
    offsets = {'SOFT': 0.0, 'MEDIUM': 0.5, 'HARD': 1.0}
    for c in ['SOFT', 'MEDIUM', 'HARD']:
        models[c] = AnchoredModel(quali_pace, deg_slopes[c], offsets[c])
    return models

# --- UI ---
st.title("🏎️ F1 Strategy: Stability Engine V4.0 (Multi-Era Adaptive)")
st.sidebar.header("Configuration")
race_options = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix", "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix", "Singapore Grand Prix", "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"]
selected_race = st.sidebar.selectbox("Track", race_options, index=21)
target_year = st.sidebar.number_input("Year", 2023, 2026, 2024)
sim_count = st.sidebar.slider("Simulations per Driver", 50, 500, 150)
sc_prob = st.sidebar.slider("Safety Car Probability", 0.0, 1.0, 0.10)

if st.button("🚀 RUN UNIVERSAL PREDICTION", type="primary"):
    round_num = get_round_from_name(target_year, selected_race)
    with st.spinner(f"📡 Fetching {target_year} Session Data..."):
        session, src = load_prediction_data(target_year, round_num, selected_race)
    
    df = session.laps.dropna(subset=['LapTime'])
    df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
    fastest_laps = df.groupby('Driver')['LapTimeSec'].min().sort_values()
    
    # --- STEP 1: DYNAMIC LEARNING CONTEXT (The ASI) ---
    # We map every driver to their best time to build a hierarchy on the fly
    session_context = fastest_laps.to_dict()

    if hasattr(session, 'official_grid') and session.official_grid:
        grid_drivers = sorted(session.official_grid.keys(), key=lambda x: session.official_grid[x])
    else:
        grid_drivers = fastest_laps.head(20).index.tolist()

    total_laps = int(session.total_laps) if hasattr(session, 'total_laps') else 58
    field_data = calculate_field_deltas(session)
    
    # Extract enhanced features
    with st.spinner("🧠 Analyzing driver histories and track characteristics..."):
        track_chars = extract_track_characteristics(target_year, selected_race)
        st.sidebar.info(f"📊 Track Overtaking: {track_chars['overtaking_difficulty']:.2f} | Quali Impact: {track_chars['qualifying_importance']:.2f}")

    results_agg = []
    progress = st.progress(0)
    for i, driver in enumerate(grid_drivers):
        models = train_anchored_models(session, driver, field_data, fastest_laps)
        strat_name, strat_laps, _ = optimize_strategy(models, total_laps)
        
        if "1-Stop" in strat_name:
            strat_list = [('SOFT', strat_laps), ('HARD', total_laps - strat_laps)]
        else:
            strat_list = [('SOFT', strat_laps[0]), ('MEDIUM', strat_laps[1]-strat_laps[0]), ('HARD', total_laps - strat_laps[1])]
            
        start_pos = i + 1
        
        # Enhanced feature extraction
        form_index, _, confidence = get_driver_recent_form(driver, target_year, round_num)
        driver_features, _ = extract_driver_historical_features(driver, target_year, round_num)
        compound_affinity = extract_driver_compound_affinity(driver, target_year, round_num)
        
        times, hists = [], []
        for _ in range(sim_count):
            hist, status = simulate_race_history(
                models, strat_list, total_laps, 
                grid_position=start_pos, consistencies=0.12, 
                driver_form=form_index, safety_car_prob=sc_prob,
                driver_code=driver,
                session_context=session_context,
                driver_features=driver_features,
                track_characteristics=track_chars,
                compound_affinity=compound_affinity
            )
            times.append(hist[-1])
            hists.append(hist)

        avg_t = np.mean(times)
        results_agg.append({
            'Driver': driver, 'AvgTime': avg_t, 'Range': np.ptp(times), 
            'RepHistory': min(hists, key=lambda h: abs(h[-1] - avg_t)),
            'Strategy': strat_name, 'Strat_laps': strat_laps, 'StartPos': start_pos,
            'RawTimes': times
        })
        progress.progress((i+1)/len(grid_drivers))

    results_agg = sorted(results_agg, key=lambda x: x['AvgTime'])
    
    # --- AUDIT & PERFORMANCE DIAGNOSTICS ---
    st.divider()
    st.header("🔍 Universal Accuracy Audit (Superman Score)")
    audit = evaluate_sim_performance(target_year, selected_race, results_agg)
    
    if audit:
        # Calculate Rank Error dynamically
        audit['data']['RankError'] = (audit['data']['Predicted'] - audit['data']['Actual']).abs()
        avg_error = audit['data']['RankError'].mean()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Spearman Correlation", f"{audit['correlation']:.2f}")
        c2.metric("Podium Hits", f"{audit['podium_accuracy']}/3")
        c3.metric("Avg Rank Error", f"{avg_error:.1f} Pos")

        # Dynamic Calibration Report
        with st.sidebar:
            st.divider()
            st.subheader("🛠️ Strategy Calibration")
            outliers = audit['data'][audit['data']['RankError'] >= 5]
            if not outliers.empty:
                st.warning(f"Found {len(outliers)} Delta Outliers.")
                for _, row in outliers.iterrows():
                    st.write(f"**{row['Driver']}**: Model P{row['Predicted']} | Actual P{row['Actual']}")
            else:
                st.success("Universal Hierarchy is stable.")

        st.dataframe(audit['data'].sort_values('Actual'), use_container_width=True)
        if audit['correlation'] > 0.8: st.balloons()

    # Visualizing Finish Distributions
    st.plotly_chart(go.Figure(data=[go.Violin(y=[t for t in r['RawTimes']], name=r['Driver'], box_visible=True, line_color=get_driver_color(r['Driver'])) for r in results_agg[:10]], layout=dict(template="plotly_dark", title="Pace Distribution (Monte Carlo Finish Times)")))