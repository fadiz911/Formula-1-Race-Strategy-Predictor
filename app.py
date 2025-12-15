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
        
        if prediction.size == 1:
            return float(prediction.item())
        else:
            return prediction.astype(float)

# NEW LOCATION FOR FORMAT_STRATEGY (BEFORE IT'S CALLED)
def format_strategy(strat_name, strat_laps, total_laps):
    """Converts the optimized strategy output into a readable string format (e.g., 'S 15 / H 42')."""
    if strat_name == "1-Stop":
        l1 = strat_laps
        l2 = total_laps - l1
        return f"S {l1} / H {l2}"
    elif strat_name == "2-Stop":
        l1, l2_end = strat_laps
        l2_len = l2_end - l1
        l3_len = total_laps - l2_end
        return f"S {l1} / H {l2_len} / S {l3_len}"
    return "N/A"

def get_round_from_name(year, race_name):
    try:
        schedule = fastf1.get_event_schedule(year)
        key = race_name.replace(" Grand Prix", "").strip()
        if "Emilia" in key: key = "Emilia"
        race = schedule[schedule['EventName'].str.contains(key, case=False, na=False)]
        if not race.empty: return int(race.iloc[0]['RoundNumber'])
    except: pass
    map_modern = {"Bahrain":1, "Saudi":2, "Australia":3, "Japan":4, "China":5, "Miami":6, "Emilia":7, "Monaco":8, "Canada":9, "Spain":10, "Austria":11, "Britain":12, "Hungary":13, "Belgium":14, "Netherlands":15, "Italy":16, "Azerbaijan":17, "Singapore":18, "United":19, "Mexico":20, "Brazil":21, "Las":22, "Qatar":23, "Abu":24}
    for k, v in map_modern.items():
        if k in race_name: return v
    return None

def train_anchored_models(session, driver, field_data, fastest_laps):
    try: quali_pace = fastest_laps[driver]
    except: quali_pace = fastest_laps.median()
    deg_slopes = {'SOFT': 0.12, 'MEDIUM': 0.08, 'HARD': 0.05}
    found_real = False
    try:
        laps = get_driver_laps(session, driver)
        laps = filter_practice_long_runs(laps)
        if not laps.empty:
            for c in ['SOFT', 'MEDIUM', 'HARD']:
                model = train_tire_model(laps, c)
                if model:
                    t1 = model.predict([[1]]); t2 = model.predict([[2]]) 
                    deg_slopes[c] = max(0.0, t2 - t1)
                    found_real = True
    except: pass
    if not found_real:
        var = random.uniform(0.9, 1.1)
        for c in deg_slopes: deg_slopes[c] *= var
    offsets = {'SOFT': 0.0, 'MEDIUM': 0.5, 'HARD': 1.0}
    models = {}
    for c in ['SOFT', 'MEDIUM', 'HARD']:
        models[c] = AnchoredModel(quali_pace, deg_slopes[c], offsets[c])
    return models

# --- UI ---
st.title("🏎️ F1 Strategy: Reactive Race Engine")
st.sidebar.header("Configuration")
race_options = ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix", "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix", "Singapore Grand Prix", "United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix", "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"]
selected_race = st.sidebar.selectbox("Track", race_options, index=21)
target_year = st.sidebar.number_input("Year", 2023, 2026, 2024)
sim_count = st.sidebar.slider("Simulations per Driver", 50, 1000, 200)

# Safety Car Configuration
st.sidebar.markdown("---")
st.sidebar.header("Safety Car Risk")
SC_PROB_MAP = {
    'Low (Bahrain/Spain)': 0.15,
    'Medium (Suzuka/Silverstone)': 0.30,
    'High (Monza/Baku)': 0.50,
    'Extreme (Monaco/Singapore)': 0.80
}
sc_risk_selection = st.sidebar.selectbox(
    "Safety Car Probability", 
    list(SC_PROB_MAP.keys()), 
    index=1
)
sc_prob = SC_PROB_MAP[sc_risk_selection]

# GRID OVERRIDE (Optional)
with st.sidebar.expander("🛠️ Manual Grid Override"):
    st.info("Leave 0 to use Official Grid.")
    grid_overrides = {}
    c1, c2 = st.columns(2)
    with c1:
        grid_overrides['VER'] = st.number_input("VER", 0, 20, 0)
        grid_overrides['NOR'] = st.number_input("NOR", 0, 20, 0)
    with c2:
        grid_overrides['LEC'] = st.number_input("LEC", 0, 20, 0)
        grid_overrides['HAM'] = st.number_input("HAM", 0, 20, 0)

if st.button("🚀 RUN PREDICTION ENGINE", type="primary"):
    
    round_num = get_round_from_name(target_year, selected_race)
    if not round_num: st.error("Race not found."); st.stop()
    
    # 1. LOAD DATA
    found_session = None
    with st.spinner(f"📡 Downloading & Normalizing Session Data..."):
        session, src = load_prediction_data(target_year, round_num, selected_race)
        if hasattr(session, 'laps'): found_session = session

    if not found_session: st.error("No Data Found. Try another race/year."); st.stop()

    # 2. SETUP GRID (SMART AUTO-DETECT)
    df = found_session.laps
    df = df.dropna(subset=['LapTime'])
    
    practice_drivers = df['Driver'].unique().tolist()
    
    df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
    fastest_laps = df.groupby('Driver')['LapTimeSec'].min().sort_values()
    
    grid_drivers = []
    
    # A. Check for Official Grid
    if hasattr(found_session, 'official_grid') and found_session.official_grid:
        official_grid = found_session.official_grid
        grid_drivers = sorted(official_grid.keys(), key=lambda x: official_grid[x])
        grid_drivers = [d for d in grid_drivers if d in practice_drivers]
        if grid_drivers:
            st.success(f"✅ Loaded Official Qualifying Grid (Pole: {grid_drivers[0]})")
        else:
             st.warning("⚠️ Official Grid found, but no matching practice data.")
             grid_drivers = fastest_laps.head(22).index.tolist()
        
    # B. Fallback to Practice Speed
    else:
        st.warning("⚠️ Qualifying Data Unavailable. Using Practice Pace for Grid.")
        grid_drivers = fastest_laps.head(22).index.tolist()
        
    # C. Force Include Stars (Safety Net)
    stars = ['VER', 'NOR', 'LEC', 'HAM', 'PIA', 'RUS', 'SAI']
    for star in stars:
        if star not in grid_drivers and star in practice_drivers:
            grid_drivers.append(star)
            
    grid_drivers = list(dict.fromkeys(grid_drivers))
    
    field_data = calculate_field_deltas(found_session)
    try: total_laps = int(found_session.total_laps)
    except: total_laps = 57

    # 3. ANALYZE FORM
    form_data = {}
    with st.status("📊 Analyzing Recent Driver Form (Last 5 Races)...", expanded=True) as status:
        cols = st.columns(5)
        for i, driver in enumerate(grid_drivers[:10]):
             with cols[i%5]:
                form_index, races = get_driver_recent_form(driver, target_year, round_num)
                form_data[driver] = form_index
                
                d_pct = (1.0 - form_index) * 100
                color = "green" if d_pct > 0 else "red"
                st.markdown(f"**{driver}**: :{color}[{d_pct:+.1f}%]")
                
        for driver in grid_drivers[10:]:
            f, _ = get_driver_recent_form(driver, target_year, round_num)
            form_data[driver] = f
            
        status.update(label="✅ Form Analysis Complete", state="complete", expanded=False)

    # 4. RUN MONTE CARLO
    results_agg = []
    progress_bar = st.progress(0)
    
    st.write(f"Running {sim_count} simulations per driver...")
    
    for i, driver in enumerate(grid_drivers):
        # Train & Optimize Strategy
        models = train_anchored_models(found_session, driver, field_data, fastest_laps)
        strat_name, strat_laps, _ = optimize_strategy(models, total_laps)
        
        if strat_name == "1-Stop": 
            strat_list = [('SOFT', strat_laps), ('HARD', total_laps - strat_laps)]
        else: 
            l1 = strat_laps[0]
            strat_list = [('SOFT', l1), ('HARD', strat_laps[1]-l1), ('SOFT', total_laps-strat_laps[1])]
            
        # Determine Start Pos (Override > Official > Practice Index)
        start_pos = i + 1
        if hasattr(found_session, 'official_grid') and driver in found_session.official_grid:
            start_pos = found_session.official_grid[driver]
        if driver in grid_overrides and grid_overrides[driver] > 0:
            start_pos = grid_overrides[driver]
            
        driver_times = []
        driver_hists = []
        driver_raw_times = [] # To store all times for the violin plot
        
        # Convergence Variables
        prev_avg = 0
        stable_count = 0
        
        # SIMULATION LOOP (With SC & Reactive Physics)
        for sim_idx in range(sim_count):
            hist, run_status = simulate_race_history(
                models, strat_list, total_laps, 
                grid_position=start_pos, 
                consistencies=0.15, 
                pit_loss_avg=22.5,
                driver_form=form_data.get(driver, 1.0),
                safety_car_prob=sc_prob
            )
            
            # Record time, penalizing DNF heavily
            if "DNF" not in run_status:
                driver_times.append(hist[-1])
                driver_hists.append(hist)
                driver_raw_times.append(hist[-1]) # Store for distribution plot
            else:
                # FIX: Append a DNF placeholder to driver_hists to maintain parallel lists
                penalty_time = 99999.0 + random.uniform(0, 10)
                driver_times.append(penalty_time)
                driver_hists.append([]) 
                driver_raw_times.append(penalty_time) # Store penalty time in raw list

            # Check convergence
            if sim_idx > 50 and sim_idx % 50 == 0:
                current_avg = np.mean(driver_times)
                if abs(current_avg - prev_avg) < 0.05: 
                    stable_count += 1
                else: stable_count = 0
                prev_avg = current_avg
                if stable_count >= 3: break
        
        # Stats Aggregation
        finished_times = [t for t in driver_times if t < 90000.0]
        
        if finished_times:
            avg_time = np.mean(finished_times)
            min_time = np.min(finished_times)
            max_time = np.max(finished_times)
            
            # Find representative history
            valid_hists = [h for i, h in enumerate(driver_hists) if driver_times[i] < 90000.0]

            if valid_hists:
                rep_hist = min(valid_hists, key=lambda h: abs(h[-1] - avg_time))
            else:
                rep_hist = []
        else:
            avg_time = 99999.0
            min_time = 99999.0
            max_time = 99999.0
            rep_hist = []

        results_agg.append({
            'Driver': driver,
            'AvgTime': avg_time,
            'Range': max_time - min_time,
            'SimsRun': len(driver_times),
            'RepHistory': rep_hist,
            'Strategy': strat_name,
            'Strat_laps': strat_laps,
            'StartPos': start_pos,
            'DNF_Rate': (len(driver_times) - len(finished_times)) / len(driver_times) if len(driver_times) > 0 else 0,
            'RawTimes': driver_raw_times # NEW: Store all raw times for Violin Plot
        })
        progress_bar.progress((i+1)/len(grid_drivers))

    # 5. RESULTS DISPLAY
    results_agg = sorted(results_agg, key=lambda x: x['AvgTime'])
    winner = results_agg[0]
    
    st.divider()
    
    # --- 5A. PREDICTED WINNER METRICS ---
    col1, col2, col3 = st.columns([1.5, 1, 1])
    
    with col1:
        st.subheader(f"🏆 Predicted Winner: {winner['Driver']}")
        # FIX: format_strategy is now available
        st.markdown(f"**Strategy:** `{format_strategy(winner['Strategy'], winner['Strat_laps'], total_laps)}`")
    
    with col2:
        st.metric("Avg Race Time", f"{winner['AvgTime'] / 3600:.2f} hours")
    
    with col3:
        st.metric("Race Volatility (Range)", f"±{winner['Range']/2:.1f}s")
        st.caption(f"Based on {winner['SimsRun']} runs.")

    # --- 5B. MAIN TABLE with Strategy Details ---
    st.subheader("🏁 Predictive Leaderboard & Strategy Summary")
    
    display_data = []
    for r in results_agg:
        gap = r['AvgTime'] - winner['AvgTime']
        
        display_data.append({
            "Driver": r['Driver'],
            "Pred Pos": results_agg.index(r) + 1,
            "Start": r['StartPos'],
            "Strategy": format_strategy(r['Strategy'], r['Strat_laps'], total_laps),
            "Gap": f"+{gap:.1f}s",
            "Form": f"{(1.0 - form_data.get(r['Driver'], 1.0))*100:+.1f}%",
            "DNF Chance": f"{r['DNF_Rate'] * 100:.1f}%"
        })
    st.dataframe(pd.DataFrame(display_data), use_container_width=True)
    
    st.markdown("---")
    st.header("📊 Detailed Race Visualizations")
    
    chart_col1, chart_col2 = st.columns(2)

    # --- 5C. RACE TIME DISTRIBUTION (Violin Plot) ---
    with chart_col1:
        st.subheader("Time Volatility (Monte Carlo Results)")
        
        plot_data = []
        for r in results_agg[:10]:
            # Use the actual raw data and filter out penalty times
            data = [t for t in r['RawTimes'] if t < 90000.0]

            for t in data:
                plot_data.append({'Driver': r['Driver'], 'Time (s)': t})

        df_plot = pd.DataFrame(plot_data)
        
        fig_dist = go.Figure()
        
        for driver in df_plot['Driver'].unique():
            driver_data = df_plot[df_plot['Driver'] == driver]['Time (s)']
            
            fig_dist.add_trace(go.Violin(
                y=driver_data,
                name=driver,
                box_visible=True,
                meanline_visible=True,
                line_color=get_driver_color(driver),
                fillcolor=get_driver_color(driver),
                opacity=0.6
            ))
            
        fig_dist.update_layout(
            template="plotly_dark", 
            title="Distribution of Predicted Race Times",
            yaxis_title="Finish Time (Seconds)",
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)


    # --- 5D. DNF/RISK BAR CHART ---
    with chart_col2:
        st.subheader("DNF & Strategy Risk")
        risk_data = pd.DataFrame(results_agg)
        risk_data['DriverColor'] = risk_data['Driver'].apply(get_driver_color)
        
        fig_risk = go.Figure(data=[
            go.Bar(
                x=risk_data['Driver'].head(10), 
                y=risk_data['DNF_Rate'].head(10) * 100,
                marker_color=risk_data['DriverColor'].head(10),
                opacity=0.8
            )
        ])
        fig_risk.update_layout(
            template="plotly_dark",
            title="Driver DNF Probability",
            yaxis_title="DNF Chance (%)",
            xaxis_title="",
            showlegend=False
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # --- 5E. RACE TRACE (Consistent Driver Colors) ---
    st.subheader("📉 Race Trace (Median Scenario)")
    
    fig = go.Figure()
    winner_hist = winner['RepHistory']
    
    for r in results_agg[:10]: 
        driver_code = r['Driver']
        hist = r['RepHistory']
        
        if hist: 
            driver_color = get_driver_color(driver_code) 
            
            gaps = [winner_hist[j] - hist[j] for j in range(min(len(hist), len(winner_hist)))]
            
            fig.add_trace(go.Scatter(
                y=gaps, 
                mode='lines', 
                name=driver_code, 
                line=dict(color=driver_color, width=3)
            ))
            
    fig.update_layout(template="plotly_dark", height=600, xaxis_title="Lap", yaxis_title="Gap to Leader (s)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)