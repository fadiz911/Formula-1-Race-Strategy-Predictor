import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import logging
import random

# Import core predictor files
from src.lstm_predictor import LSTMPredictor
from src.transformer_predictor import TransformerPredictor
from src.ml_predictor import MLPredictor
from src.ensemble_predictor import EnsemblePredictor
from src.predictor import load_prediction_data, get_driver_recent_form
from src.predictor import train_anchored_models as main_train_anchored_models
from src.strategy_engine import calculate_field_deltas
from src.optimizer import optimize_strategy
from src.simulation import simulate_race_history
from src.feature_engineering import (
    extract_driver_historical_features,
    extract_track_characteristics,
    extract_driver_compound_affinity,
)
from src.reliability_model import extract_reliability_profile

# Suppress logs
logging.getLogger('fastf1').setLevel(logging.ERROR)

# Page config
st.set_page_config(
    page_title="F1 Race Strategy Predictor & Simulator",
    page_icon="🏎️",
    layout="wide"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1e27; padding: 18px; border-radius: 12px; border: 1px solid #2e3748; }
    h1 { color: #ff1801; font-family: 'Outfit', sans-serif; font-weight: 800; }
    h2, h3 { color: #00ffd0; font-family: 'Outfit', sans-serif; }
    .stButton>button {
        background-color: #ff1801;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #cc1200;
        transform: scale(1.02);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1e27;
        border: 1px solid #2e3748;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #a0aec0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff1801 !important;
        color: white !important;
        border: 1px solid #ff1801 !important;
    }
</style>
""", unsafe_allow_html=True)

# Team colors for visualization
TEAM_COLORS = {
    'Red Bull Racing': '#0600ef',
    'Mercedes': '#00d2be',
    'Ferrari': '#dc0000',
    'McLaren': '#ff8700',
    'Aston Martin': '#006f62',
    'Alpine': '#0090ff',
    'Williams': '#005aff',
    'AlphaTauri': '#2b4562',
    'Alfa Romeo': '#900000',
    'Haas F1 Team': '#ffffff',
    'Sauber': '#00e701',
    'Kick Sauber': '#00e701',
    'RB': '#2b4562'
}

# Load Models
@st.cache_resource
def load_lstm_model():
    return LSTMPredictor()

@st.cache_resource
def load_transformer_model():
    return TransformerPredictor()

@st.cache_resource
def load_ensemble_model():
    return EnsemblePredictor()

def format_seconds_to_time(sec):
    if sec >= 90000.0:
        return "DNF"
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    milliseconds = int((sec % 1) * 1000)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"

def format_timedelta(td):
    if pd.isnull(td):
        return "N/A"
    total_seconds = td.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    elif minutes > 0:
        return f"{minutes}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{seconds}.{milliseconds:03d}"

def format_strategy(strat_name, strat_laps, total_laps):
    if "1-Stop" in strat_name:
        l1 = int(strat_laps)
        l2 = int(total_laps - l1)
        return f"S {l1} / H {l2}"
    elif "2-Stop" in strat_name:
        l1, l2_end = strat_laps
        l2_len = int(l2_end - l1)
        l3_len = int(total_laps - l2_end)
        return f"S {int(l1)} / H {l2_len} / S {l3_len}"
    return "N/A"

def get_race_predictions(predictor, year, round_num, selected_track):
    """Get predictions for all drivers in a race"""
    df = predictor.raw_history_df.copy()
    if df.empty:
        return None
    
    # Filter after unscaling Year/Round
    df = df[
        (df['Year'].round().astype(int) == year) & 
        (df['Round'].round().astype(int) == round_num)
    ].copy()
    
    if df.empty:
        return None

    # Load actual race results to get Time and Fastest Lap
    race_results = {}
    actual_total_laps = None
    try:
        race_session = fastf1.get_session(year, round_num, 'R')
        race_session.load(laps=True, telemetry=False, weather=False, messages=False)
        if hasattr(race_session, 'laps') and not race_session.laps.empty:
            actual_total_laps = int(race_session.laps['LapNumber'].max())
        winner_time_td = race_session.results.iloc[0]['Time']
        for _, r_row in race_session.results.iterrows():
            d_abbr = r_row['Abbreviation']
            # Get fastest lap
            d_laps = race_session.laps.pick_drivers(d_abbr)
            fastest_lap_td = d_laps.pick_fastest()['LapTime'] if not d_laps.empty else pd.NaT
            
            # Format time/status as gap relative to winner
            status = r_row['Status']
            if r_row['Position'] == 1:
                gap_str = "Leader"
            elif status == 'Finished':
                # FastF1 already stores the relative gap to the winner in Time for positions > 1
                gap_str = f"+{format_timedelta(r_row['Time'])}s"
            else:
                gap_str = status
                
            race_results[d_abbr] = {
                'ActualGap': gap_str,
                'FastestLap': format_timedelta(fastest_lap_td)
            }
    except Exception as e:
        pass

    # Load practice session data for simulation-based predicted times
    session = None
    try:
        session, src = load_prediction_data(year, round_num, selected_track)
        if hasattr(session, 'laps') and not session.laps.empty:
            if actual_total_laps is not None:
                session.total_laps = actual_total_laps
            df_laps = session.laps.dropna(subset=['LapTime']).copy()
            df_laps['LapTimeSec'] = df_laps['LapTime'].dt.total_seconds()
            fastest_laps = df_laps.groupby('Driver')['LapTimeSec'].min().sort_values()
            field_data = calculate_field_deltas(session)
            total_laps = int(session.total_laps) if hasattr(session, 'total_laps') else 57
            session_context = fastest_laps.to_dict()
            track_chars = extract_track_characteristics(year, selected_track)
        else:
            session = None
    except Exception as e:
        session = None
    
    # Detect wet weather
    wet_weather = bool(df['IsWet'].iloc[0] > 0) if (not df.empty and 'IsWet' in df.columns) else False
    
    feature_cols = predictor.scaler.feature_names_in_
    predictions = []
    for _, row in df.iterrows():
        # Build context from scaled/unscaled fields
        ctx = {}
        for col in feature_cols:
            ctx[col] = row[col]
            
        pred_scaled = predictor.predict_position(
            driver=row['Driver'],
            year=year,
            round_num=round_num,
            current_grid=row['GridPos'],
            current_team_name=row['Team'],
            current_track_name=row['Track'],
            context=ctx
        )
        
        if pred_scaled is not None:
            # Unscale prediction (FinishPos is always scaled at index 0 or similar)
            # Both LSTM and Transformer now use scaled targets.
            pred_pos = (pred_scaled * predictor.scaler.scale_[0]) + predictor.scaler.mean_[0]
            
            d_abbr = row['Driver']
            actual_gap = race_results.get(d_abbr, {}).get('ActualGap', 'N/A')
            fastest_lap = race_results.get(d_abbr, {}).get('FastestLap', 'N/A')
            
            # Extract reliability profile
            profile = extract_reliability_profile(d_abbr, year, round_num)
            rel_risk = profile.get('reliability_risk', 0.0)
            if rel_risk < 0.10:
                dnf_risk_label = "Low"
            elif rel_risk < 0.25:
                dnf_risk_label = "Medium"
            else:
                dnf_risk_label = "High"
                
            # Blend reliability risk into finishing rank sorting
            sort_score = pred_pos + (rel_risk * 2.0)
            
            # Predict race time via deterministic simulation
            predicted_time_seconds = 99999.0
            if session is not None:
                try:
                    s_pos = int(row['GridPos'])
                    models = main_train_anchored_models(session, d_abbr, field_data, fastest_laps)
                    strat_name, strat_laps, _ = optimize_strategy(models, total_laps)
                    
                    if "1-Stop" in strat_name:
                        strat_list = [('SOFT', int(strat_laps)), ('HARD', int(total_laps - int(strat_laps)))]
                    else:
                        l1, l2 = strat_laps
                        strat_list = [('SOFT', int(l1)), ('MEDIUM', int(l2 - l1)), ('HARD', int(total_laps - l2))]
                    
                    track_lap_time = fastest_laps.median() if not fastest_laps.empty else 80.0
                    corr = predictor.get_correction_factor(
                        d_abbr, s_pos, row['Team'], row['Track'], year, round_num, 10,
                        track_lap_time=track_lap_time
                    )
                    
                    form_idx, _, _ = get_driver_recent_form(d_abbr, year, round_num)
                    driver_features, _ = extract_driver_historical_features(d_abbr, year, round_num)
                    compound_affinity = extract_driver_compound_affinity(d_abbr, year, round_num)
                    
                    sc_prob = track_chars.get('safety_car_likelihood', 0.15) if track_chars else 0.15
                    hist, _ = simulate_race_history(
                        models, strat_list, total_laps,
                        grid_position=s_pos,
                        consistencies=0.0,
                        driver_form=form_idx,
                        safety_car_prob=sc_prob,
                        driver_code=d_abbr,
                        session_context=session_context,
                        driver_features=driver_features,
                        track_characteristics=track_chars,
                        compound_affinity=compound_affinity,
                        ml_correction=corr,
                        reliability_mode='conservative',
                        wet_weather=wet_weather
                    )
                    predicted_time_seconds = hist[-1]
                except Exception as e:
                    pass
            
            predictions.append({
                'Driver': d_abbr,
                'Team': row['Team'],
                'Grid': int(row['GridPos']),
                'Predicted': round(np.clip(pred_pos, 1, 20), 1),
                'SortScore': sort_score,
                'Actual': int(row['FinishPos']),
                'ActualGap': actual_gap,
                'FastestLap': fastest_lap,
                'PredictedTimeSeconds': predicted_time_seconds,
                'DNF Risk': dnf_risk_label
            })
            
    if not predictions:
        return None
    df_res = pd.DataFrame(predictions).sort_values('SortScore')
    df_res['PredictedRank'] = range(1, len(df_res) + 1)
    df_res['Error'] = (df_res['PredictedRank'] - df_res['Actual']).abs()
    
    # Calculate predicted gaps from simulated times (PredictedTimeSeconds) relative to predicted winner
    pred_winner_row = df_res[df_res['PredictedRank'] == 1]
    if not pred_winner_row.empty and pred_winner_row.iloc[0]['PredictedTimeSeconds'] < 90000.0:
        pred_winner_time = pred_winner_row.iloc[0]['PredictedTimeSeconds']
        df_res['Predicted Gap'] = df_res['PredictedTimeSeconds'].apply(
            lambda x: "DNF" if x >= 90000.0 else (
                "Leader" if abs(x - pred_winner_time) < 0.001 else f"+{x - pred_winner_time:.1f}s"
            )
        )
    else:
        df_res['Predicted Gap'] = 'N/A'
    return df_res

def main():
    st.title("🏎️ Formula 1 Race Strategy Predictor & Simulator")
    
    # Sidebar config
    st.sidebar.title("🛠️ Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Select Prediction Architecture",
        [
            "Ensemble (LSTM + Transformer) - Max Accuracy",
            "Transformer (V5) - Telemetry Enabled", 
            "LSTM (V4) - Temporal Sequence"
        ]
    )
    
    if "Ensemble" in model_choice:
        predictor = load_ensemble_model()
        model_name = "Ensemble (70% LSTM + 30% Trans)"
        st.sidebar.success("🤖 Active Model: Ensemble Blend")
    elif "Transformer" in model_choice:
        predictor = load_transformer_model()
        model_name = "Transformer V5"
        st.sidebar.success("🤖 Active Model: Transformer (V5)")
    else:
        predictor = load_lstm_model()
        model_name = "LSTM V4"
        st.sidebar.success("🤖 Active Model: LSTM (V4)")
        
    # Sidebar metrics
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Validation Stats")
    if "Ensemble" in model_choice:
        st.sidebar.metric("Spearman Correlation (2024)", "0.982", "+0.8% vs Trans")
        st.sidebar.metric("Spearman Correlation (2025)", "0.988", "+2.1% vs Trans")
    elif "Transformer" in model_choice:
        st.sidebar.metric("Spearman Correlation (2024)", "0.974", "V5 Upgraded")
        st.sidebar.metric("Spearman Correlation (2025)", "0.967", "V5 Upgraded")
    else:
        st.sidebar.metric("Spearman Correlation (2024)", "0.978", "V4 Baseline")
        st.sidebar.metric("Spearman Correlation (2025)", "0.987", "V4 Baseline")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Features")
    if "Ensemble" in model_choice:
        st.sidebar.markdown("""
        - Blended Sequence History
        - Physical & Telemetry Fusion
        - Advanced Feature Weighting
        """)
    elif "Transformer" in model_choice:
        st.sidebar.markdown("""
        - Driver Consistency Score
        - Track Performance History
        - Qualifying teammates delta
        - Practice Pace comparison
        - Team Rolling Average Points
        - Driver Reliability (DNF) Risk
        """)
    else:
        st.sidebar.markdown("""
        - Rolling Consistency
        - Track-specific History
        - Qualifying delta (teammate)
        - Practice Pace diff
        - Weather / Wet Flag
        """)

    # Year & GP Selection
    col_yr, col_gp = st.columns(2)
    with col_yr:
        year = st.selectbox("Select Season", [2023, 2024, 2025, 2026], index=3)
        
    if predictor.history_df.empty:
        st.error("❌ Model history failed to load. FastF1 download might have timed out.")
        return
        
    races = predictor.raw_history_df[predictor.raw_history_df['Year'] == year][['Round', 'Track']].drop_duplicates().sort_values('Round')
    race_options = {row['Track']: row['Round'] for _, row in races.iterrows()}
    
    with col_gp:
        selected_track = st.selectbox("Select Grand Prix", list(race_options.keys()))
        
    round_num = race_options[selected_track]
    
    # App Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔮 Single Race Prediction", 
        "🎲 Monte Carlo Simulation", 
        "📈 Advanced Visualizations"
    ])
    
    # Session state for sharing results across tabs
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = None
    if 'last_run' not in st.session_state:
        st.session_state.last_run = None
        
    # --- TAB 1: SINGLE RACE PREDICTION ---
    with tab1:
        st.header(f"🔮 Direct Model Predictions: {year} {selected_track}")
        st.write(f"This runs a direct sequence-based forward pass using the **{model_name}** model.")
        
        if st.button("Predict Finishing Order", key="btn_predict", type="primary"):
            with st.spinner("Generating predictions..."):
                results = get_race_predictions(predictor, year, round_num, selected_track)
                st.session_state.results = results
                st.session_state.last_run = "prediction"
                
        if st.session_state.results is not None:
            results = st.session_state.results
            
            # Key metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            mae = results['Error'].mean()
            correlation = results[['Predicted', 'Actual']].corr().iloc[0, 1]
            max_error = results['Error'].max()
            perfect = (results['Error'] < 1).sum()
            
            m_col1.metric("Mean Error (MAE)", f"{mae:.2f} pos")
            m_col2.metric("Correlation", f"{correlation:.3f}")
            m_col3.metric("Max Error", f"{max_error:.1f} pos")
            m_col4.metric("±1 Position Accuracy", f"{perfect} drivers")
            
            # Display Table
            st.subheader("Predicted Finishing Order")
            display_df = results.copy()
            if 'ActualGap' not in display_df.columns:
                display_df['ActualGap'] = 'N/A'
            if 'Predicted Gap' not in display_df.columns:
                display_df['Predicted Gap'] = 'N/A'
            if 'FastestLap' not in display_df.columns:
                display_df['FastestLap'] = 'N/A'
            if 'DNF Risk' not in display_df.columns:
                display_df['DNF Risk'] = 'Low'
                
            display_df['Predicted Pos'] = display_df['PredictedRank'].apply(lambda x: f"P{x}")
            display_df['Actual Pos'] = display_df['Actual'].apply(lambda x: f"P{x}")
            display_df['Grid Pos'] = display_df['Grid'].apply(lambda x: f"P{x}")
            display_df['Accuracy'] = display_df['Error'].apply(lambda x: f"±{int(x)}")
            
            display_df.rename(columns={
                'ActualGap': 'Actual Gap',
                'Predicted Gap': 'Predicted Gap',
                'FastestLap': 'Fastest LapTime'
            }, inplace=True)
            
            st.dataframe(
                display_df[['Driver', 'Team', 'Grid Pos', 'Predicted Pos', 'Predicted Gap', 'Actual Pos', 'Actual Gap', 'Fastest LapTime', 'DNF Risk', 'Accuracy']],
                use_container_width=True,
                hide_index=True
            )
            
    # --- TAB 2: MONTE CARLO SIMULATION ---
    with tab2:
        st.header(f"🎲 Monte Carlo Simulation Engine")
        st.write("Simulates race laps under fuel load, tire wear, traffic, safety cars, and driver form.")
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            sim_count = st.slider("Simulation Count", 50, 300, 100, 10)
        with col_c2:
            sc_prob = st.slider("Safety Car Probability (%)", 0, 100, 30) / 100.0
        with col_c3:
            reliability_stress = st.slider("Reliability Stress Level", 0.5, 2.0, 1.0, 0.1)
            
        st.subheader("Starting Grid Configuration (Overrides)")
        st.write("Customize starting positions below before launching the simulation.")
        
        # Load race drivers to setup overrides
        race_df = predictor.raw_history_df[
            (predictor.raw_history_df['Year'] == year) & 
            (predictor.raw_history_df['Round'] == round_num)
        ].copy()
        
        grid_overrides = {}
        if not race_df.empty:
            max_grid = int(max(20, race_df['GridPos'].max()))
            drivers_list = race_df['Driver'].unique().tolist()
            # 4 columns of drivers for compact display
            cols = st.columns(5)
            for i, driver in enumerate(drivers_list):
                with cols[i % 5]:
                    default_grid = int(race_df[race_df['Driver'] == driver]['GridPos'].values[0])
                    grid_overrides[driver] = st.number_input(f"{driver} Starting Grid", 1, max_grid, default_grid)
                    
        if st.button("Run Monte Carlo Race Simulation", key="btn_sim", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Loading data
            status_text.text("Loading and normalizing GP practice telemetry...")
            session, src = load_prediction_data(year, round_num, selected_track)
            progress_bar.progress(20)
            
            if session is None or not hasattr(session, 'laps') or session.laps.empty:
                st.error("❌ Failed to load laps data for this race.")
            else:
                df_laps = session.laps
                df_laps = df_laps.dropna(subset=['LapTime'])
                df_laps['LapTimeSec'] = df_laps['LapTime'].dt.total_seconds()
                fastest_laps = df_laps.groupby('Driver')['LapTimeSec'].min().sort_values()
                
                # Fetch details
                track_chars = extract_track_characteristics(year, selected_track)
                track_chars['reliability_stress'] = reliability_stress
                field_data = calculate_field_deltas(session)
                session_context = fastest_laps.to_dict()
                
                # We prioritize grid overrides
                grid_drivers = sorted(grid_overrides.keys(), key=lambda x: grid_overrides[x])
                total_laps = int(session.total_laps) if hasattr(session, 'total_laps') else 57
                
                # Pre-calculate strategies
                status_text.text("Calibrating tire degradation curves...")
                driver_strategies = {}
                baseline_ranks = {}
                
                for i, driver in enumerate(grid_drivers):
                    models = main_train_anchored_models(session, driver, field_data, fastest_laps)
                    strat_name, strat_laps, _ = optimize_strategy(models, total_laps)
                    
                    if "1-Stop" in strat_name:
                        strat_list = [('SOFT', int(strat_laps)), ('HARD', int(total_laps - int(strat_laps)))]
                    else:
                        l1, l2 = strat_laps
                        strat_list = [('SOFT', int(l1)), ('MEDIUM', int(l2 - l1)), ('HARD', int(total_laps - l2))]
                    
                    driver_strategies[driver] = (models, strat_list, strat_name, strat_laps)
                    
                    # Run deterministic simulation for ranking
                    hist, _ = simulate_race_history(
                        models, strat_list, total_laps,
                        grid_position=10, consistencies=0.0,
                        driver_form=1.0, safety_car_prob=0.0,
                        driver_code=driver, session_context=session_context
                    )
                    baseline_ranks[driver] = hist[-1]
                
                progress_bar.progress(50)
                status_text.text("Injecting neural network bias corrections...")
                
                sorted_baseline = sorted(baseline_ranks.keys(), key=lambda x: baseline_ranks[x])
                sim_ranks = {d: idx + 1 for idx, d in enumerate(sorted_baseline)}
                
                # ML/DL corrections
                corrections = {}
                for driver in grid_drivers:
                    s_pos = grid_overrides[driver]
                    team_name = race_df[race_df['Driver'] == driver]['Team'].values[0] if driver in race_df['Driver'].values else "Unknown"
                    
                    corr = predictor.get_correction_factor(
                        driver, s_pos, team_name, selected_track, year, round_num, sim_ranks[driver]
                    )
                    corrections[driver] = corr
                    
                progress_bar.progress(60)
                
                # Simulation Loop
                sim_agg = []
                for idx, driver in enumerate(grid_drivers):
                    status_text.text(f"Simulating driver race pace: {driver} ({idx+1}/{len(grid_drivers)})...")
                    models, strat_list, strat_name, strat_laps = driver_strategies[driver]
                    s_pos = grid_overrides[driver]
                    
                    # Form features
                    form_idx, _, _ = get_driver_recent_form(driver, year, round_num)
                    driver_features, _ = extract_driver_historical_features(driver, year, round_num)
                    compound_affinity = extract_driver_compound_affinity(driver, year, round_num)
                    
                    driver_times = []
                    dnfs = 0
                    for _ in range(sim_count):
                        hist, status = simulate_race_history(
                            models, strat_list, total_laps,
                            grid_position=s_pos,
                            consistencies=0.15,
                            driver_form=form_idx,
                            safety_car_prob=sc_prob,
                            driver_code=driver,
                            session_context=session_context,
                            driver_features=driver_features,
                            track_characteristics=track_chars,
                            compound_affinity=compound_affinity,
                            ml_correction=corrections.get(driver, 0.0)
                        )
                        if "DNF" in status:
                            dnfs += 1
                            driver_times.append(99999.0 + random.uniform(0, 10))
                        else:
                            driver_times.append(hist[-1])
                            
                    finished_times = [t for t in driver_times if t < 90000.0]
                    avg_time = np.mean(finished_times) if finished_times else 99999.0
                    
                    sim_agg.append({
                        'Driver': driver,
                        'Team': race_df[race_df['Driver'] == driver]['Team'].values[0] if driver in race_df['Driver'].values else "Unknown",
                        'AvgTime': avg_time,
                        'Strategy': format_strategy(strat_name, strat_laps, total_laps),
                        'Start': s_pos,
                        'DNF_Rate': (dnfs / sim_count) * 100
                    })
                    
                    progress_bar.progress(60 + int(40 * (idx + 1) / len(grid_drivers)))
                    
                status_text.text("Aggregating Monte Carlo simulation leaderboards...")
                sim_df = pd.DataFrame(sim_agg).sort_values('AvgTime')
                
                # Calculate gaps
                winner_time = sim_df.iloc[0]['AvgTime']
                sim_df['Gap (s)'] = sim_df['AvgTime'].apply(lambda x: f"+{x - winner_time:.1f}" if x < 90000.0 and x > winner_time else ("0.0" if x == winner_time else "DNF"))
                sim_df['Predicted Race Time'] = sim_df['AvgTime'].apply(format_seconds_to_time)
                sim_df['Pos'] = range(1, len(sim_df) + 1)
                
                st.session_state.sim_results = sim_df
                st.session_state.last_run = "simulation"
                
                status_text.success("✅ Simulation Completed!")
                
        if st.session_state.sim_results is not None:
            sim_df = st.session_state.sim_results
            
            st.subheader("🏁 Predictive Leaderboard (Average Finish Time)")
            st.dataframe(
                sim_df[['Pos', 'Driver', 'Team', 'Start', 'Strategy', 'Predicted Race Time', 'Gap (s)', 'DNF_Rate']],
                use_container_width=True,
                hide_index=True
            )
            
    # --- TAB 3: ADVANCED VISUALIZATIONS ---
    with tab3:
        st.header("📈 Interactive Race Insights")
        
        if st.session_state.last_run is None:
            st.info("💡 Please run a prediction or simulation in the previous tabs to populate charts.")
        else:
            if st.session_state.results is not None:
                results = st.session_state.results
                
                st.subheader("🎯 Prediction vs Actual Positioning")
                fig = go.Figure()
                
                # Perfect prediction diagonal
                fig.add_trace(go.Scatter(
                    x=[1, 20], y=[1, 20],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='#4a5568', dash='dash')
                ))
                
                # Markers
                fig.add_trace(go.Scatter(
                    x=results['Actual'],
                    y=results['PredictedRank'],
                    mode='markers+text',
                    text=results['Driver'],
                    textposition='top center',
                    marker=dict(
                        size=14,
                        color=[TEAM_COLORS.get(team, '#ffffff') for team in results['Team']],
                        line=dict(color='white', width=1.5)
                    ),
                    name='Drivers'
                ))
                
                fig.update_layout(
                    title=f"Accuracy mapping for {year} {selected_track} (Pearson Corr: {results[['PredictedRank', 'Actual']].corr().iloc[0, 1]:.3f})",
                    xaxis_title="Actual Race Finish",
                    yaxis_title="Model Predicted Finish",
                    height=550,
                    template="plotly_dark",
                    xaxis=dict(range=[0, 21]),
                    yaxis=dict(range=[0, 21])
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Error histogram
                st.subheader("🎯 Prediction Position Error Spread")
                fig_hist = px.histogram(
                    results,
                    x="Error",
                    nbins=12,
                    title="Distribution of errors (in positions)",
                    labels={'Error': 'Position Error magnitude'},
                    template="plotly_dark",
                    color_discrete_sequence=['#ff1801']
                )
                fig_hist.update_layout(height=350)
                st.plotly_chart(fig_hist, use_container_width=True)
                
            if st.session_state.sim_results is not None:
                sim_df = st.session_state.sim_results
                
                # Time gaps chart
                st.subheader("⏱️ Predicted Race Gaps to Winner")
                
                # Filter out DNFs
                gaps_df = sim_df[sim_df['AvgTime'] < 90000.0].copy()
                winner_time = gaps_df.iloc[0]['AvgTime']
                gaps_df['GapSeconds'] = gaps_df['AvgTime'] - winner_time
                
                fig_gaps = px.bar(
                    gaps_df,
                    x="Driver",
                    y="GapSeconds",
                    title="Time gaps (seconds) to predicted race winner",
                    color="Team",
                    color_discrete_map=TEAM_COLORS,
                    template="plotly_dark"
                )
                fig_gaps.update_layout(
                    yaxis_title="Gap (seconds)",
                    xaxis_title="Driver",
                    height=450
                )
                st.plotly_chart(fig_gaps, use_container_width=True)

if __name__ == "__main__":
    main()
