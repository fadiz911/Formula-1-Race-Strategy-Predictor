import pandas as pd
import numpy as np
import fastf1
import logging
import random
import os
import sys
import time

# --- Setup Logging and Cache ---
# Suppress noisy fastf1 logs
logging.getLogger('fastf1').setLevel(logging.ERROR)

# Ensure cache is enabled globally
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# --- Import Core Logic ---
# Ensure these files are accessible in your environment (e.g., in a 'src' directory)
try:
    from src.predictor import load_prediction_data, filter_practice_long_runs, get_driver_recent_form
    from src.strategy_engine import train_tire_model, calculate_field_deltas
    from src.optimizer import optimize_strategy
    from src.simulation import simulate_race_history
except ImportError as e:
    print(f"Error importing required modules. Ensure 'src/predictor.py', etc., exist.")
    print(f"Details: {e}")
    sys.exit(1)

# --- CONFIGURATION (Manually Set Parameters) ---
# NOTE: These match the typical defaults from your Streamlit sidebar
TARGET_YEAR = 2024
SELECTED_RACE = "Abu Dhabi Grand Prix"
SIM_COUNT = 200 # Simulations per driver
SC_PROB = 0.30 # Medium risk
GRID_OVERRIDES = {'VER': 6, 'LEC': 0} # Example: Max starts P6, others auto-detected

# --- CONSTANTS and HELPERS ---
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

def format_strategy(strat_name, strat_laps, total_laps):
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


# --- MAIN EXECUTION FUNCTION ---

def run_f1_simulation():
    start_time_total = time.time()
    print("=====================================================")
    print(f"| F1 RACE PREDICTOR: {SELECTED_RACE} {TARGET_YEAR} |")
    print(f"| Sims: {SIM_COUNT} | SC Risk: {SC_PROB*100:.0f}% |")
    print("=====================================================")
    
    round_num = get_round_from_name(TARGET_YEAR, SELECTED_RACE)
    if not round_num:
        print(f"ERROR: Could not find round number for {SELECTED_RACE}.")
        return

    # 1. LOAD DATA
    print("\n[1/4] Loading and Normalizing Data...")
    try:
        session, src = load_prediction_data(TARGET_YEAR, round_num, SELECTED_RACE)
        if not hasattr(session, 'laps'): raise Exception("No valid laps found.")
    except Exception as e:
        print(f"FATAL ERROR during data load: {e}")
        return

    df = session.laps
    df = df.dropna(subset=['LapTime'])
    practice_drivers = df['Driver'].unique().tolist()
    
    df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
    fastest_laps = df.groupby('Driver')['LapTimeSec'].min().sort_values()
    
    # 2. SETUP GRID
    grid_drivers = []
    if hasattr(session, 'official_grid') and session.official_grid:
        official_grid = session.official_grid
        grid_drivers = sorted(official_grid.keys(), key=lambda x: official_grid[x])
        grid_drivers = [d for d in grid_drivers if d in practice_drivers]
        print(f" -> Using Official Grid (Pole: {grid_drivers[0]})")
    else:
        grid_drivers = fastest_laps.head(22).index.tolist()
        print(" -> Using Practice Pace for Grid (No official Q data)")
    
    stars = ['VER', 'NOR', 'LEC', 'HAM', 'PIA', 'RUS', 'SAI']
    for star in stars:
        if star not in grid_drivers and star in practice_drivers:
            grid_drivers.append(star)
    grid_drivers = list(dict.fromkeys(grid_drivers))
    
    field_data = calculate_field_deltas(session)
    try: total_laps = int(session.total_laps)
    except: total_laps = 57
    print(f" -> Total Race Laps: {total_laps}. Analyzing {len(grid_drivers)} drivers.")

    # 3. ANALYZE FORM
    print("\n[2/4] Analyzing Driver Form (Last 5 Races)...")
    form_data = {}
    for driver in grid_drivers:
        form_index, _ = get_driver_recent_form(driver, TARGET_YEAR, round_num)
        form_data[driver] = form_index
    print(f" -> Form data calculated for {len(form_data)} drivers.")
    
    # 4. RUN MONTE CARLO
    print("\n[3/4] Running Monte Carlo Simulation...")
    results_agg = []
    
    for i, driver in enumerate(grid_drivers):
        sys.stdout.write(f"\r  -> Simulating {driver} ({i+1}/{len(grid_drivers)})...")
        sys.stdout.flush()
        
        # Train & Optimize Strategy
        models = train_anchored_models(session, driver, field_data, fastest_laps)
        strat_name, strat_laps, _ = optimize_strategy(models, total_laps)
        
        if strat_name == "1-Stop": 
            strat_list = [('SOFT', strat_laps), ('HARD', total_laps - strat_laps)]
        else: 
            l1 = strat_laps[0]
            strat_list = [('SOFT', l1), ('HARD', strat_laps[1]-l1), ('SOFT', total_laps-strat_laps[1])]
            
        # Determine Start Pos (Override > Official > Practice Index)
        start_pos = i + 1
        if driver in GRID_OVERRIDES and GRID_OVERRIDES[driver] > 0:
            start_pos = GRID_OVERRIDES[driver]
        elif hasattr(session, 'official_grid') and driver in session.official_grid:
            start_pos = session.official_grid[driver]
            
        driver_times = []
        
        # SIMULATION LOOP (Simplified Convergence Check for console)
        for sim_idx in range(SIM_COUNT):
            hist, run_status = simulate_race_history(
                models, strat_list, total_laps, 
                grid_position=start_pos, 
                consistencies=0.15, 
                pit_loss_avg=22.5,
                driver_form=form_data.get(driver, 1.0),
                safety_car_prob=SC_PROB
            )
            
            # Record time, penalizing DNF heavily
            if "DNF" not in run_status:
                driver_times.append(hist[-1])
            else:
                driver_times.append(99999.0 + random.uniform(0, 10))

        # Stats Aggregation
        finished_times = [t for t in driver_times if t < 90000.0]
        
        if finished_times:
            avg_time = np.mean(finished_times)
            min_time = np.min(finished_times)
            max_time = np.max(finished_times)
        else:
            avg_time = 99999.0
            min_time = 99999.0
            max_time = 99999.0

        results_agg.append({
            'Driver': driver,
            'AvgTime': avg_time,
            'Range': max_time - min_time,
            'Strategy': strat_name,
            'Strat_laps': strat_laps,
            'StartPos': start_pos,
            'DNF_Rate': (len(driver_times) - len(finished_times)) / len(driver_times) if len(driver_times) > 0 else 0
        })
    sys.stdout.write("\r  -> Simulation complete.\n")

    # 5. RESULTS DISPLAY
    print("\n[4/4] Generating Results...")
    results_agg = sorted(results_agg, key=lambda x: x['AvgTime'])
    winner = results_agg[0]
    
    print("\n=====================================================")
    print(f"🏆 PREDICTED WINNER: {winner['Driver']}")
    print(f"   Avg Time: {winner['AvgTime'] / 3600:.2f} hours")
    print(f"   Strategy: {format_strategy(winner['Strategy'], winner['Strat_laps'], total_laps)}")
    print("=====================================================")

    display_data = []
    for r in results_agg:
        gap = r['AvgTime'] - winner['AvgTime']
        d_pct = (1.0 - form_data.get(r['Driver'], 1.0)) * 100
        
        display_data.append({
            "Pos": results_agg.index(r) + 1,
            "Driver": r['Driver'],
            "Start": r['StartPos'],
            "Strategy": format_strategy(r['Strategy'], r['Strat_laps'], total_laps),
            "Gap (s)": f"+{gap:.1f}",
            "Form (%)": f"{d_pct:+.1f}",
            "DNF (%)": f"{r['DNF_Rate'] * 100:.1f}"
        })
        
    df_results = pd.DataFrame(display_data)
    
    # Print a nicely formatted table
    print("\nPREDICTIVE LEADERBOARD:\n")
    print(df_results.to_markdown(index=False))
    
    end_time_total = time.time()
    print(f"\nTotal Execution Time: {end_time_total - start_time_total:.2f} seconds")

if __name__ == "__main__":
    run_f1_simulation()