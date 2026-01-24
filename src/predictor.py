import fastf1
import pandas as pd
import numpy as np
import os
from src.strategy_engine import train_tire_model

# --- 1. ENABLE CACHE ---
# Create 'cache' folder if it doesn't exist and enable fastf1 caching
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

class AnchoredModel:
    """
    Optimized Linear Tire Model.
    """
    def __init__(self, quali_pace, deg_slope, compound_offset):
        self.base_pace = float(quali_pace) + float(compound_offset)
        self.deg_slope = float(deg_slope)
        
    def predict(self, X):
        # Optimized for scalar input (common in simulation loop)
        if isinstance(X, (int, float)):
             return self.base_pace + (X * self.deg_slope)
             
        # Vectorized input
        X = np.asarray(X)
        if X.ndim > 1: lap_age = X[:, 0]
        else: lap_age = X.ravel() # fast flatten
        
        return self.base_pace + (lap_age * self.deg_slope)

class CombinedSession:
    """
    A 'Virtual Session' object to hold merged practice data and the official grid map.
    """
    def __init__(self, laps, total_laps, event_name, year):
        self.laps = laps
        self.total_laps = total_laps
        self.event = {'EventName': event_name}
        self.event = {'EventName': event_name}
        self.official_grid = {} 

def train_anchored_models(session, driver, field_data, fastest_laps):
    try: quali_pace = float(fastest_laps[driver])
    except: quali_pace = float(fastest_laps.median())
    
    deg_slopes = {'SOFT': 0.12, 'MEDIUM': 0.08, 'HARD': 0.05}
    if field_data:
        # Use field averages as better defaults
        deg_slopes['MEDIUM'] = deg_slopes['SOFT'] - field_data.get('SOFT_TO_MEDIUM_DEG', 0.04)
        deg_slopes['HARD'] = deg_slopes['MEDIUM'] - field_data.get('MEDIUM_TO_HARD_DEG', 0.03)

    try:
        # Extract laps directly from session
        laps = session.laps
        driver_laps = laps[laps['Driver'] == driver]
        long_runs = filter_practice_long_runs(driver_laps)
        
        if not long_runs.empty:
            for c in ['SOFT', 'MEDIUM', 'HARD']:
                model = train_tire_model(long_runs, c)
                if model:
                    # Predict scalar tire age 1 and 2
                    t1 = model.predict([[1]])[0] 
                    t2 = model.predict([[2]])[0]
                    deg_slopes[c] = max(0.0, float(t2 - t1))
    except Exception as e:
        pass
        
    models = {}
    offsets = {'SOFT': 0.0, 'MEDIUM': 0.5, 'HARD': 1.0}
    
    if field_data:
         offsets['MEDIUM'] = field_data.get('SOFT_TO_MEDIUM_SPEED', 0.5)
         offsets['HARD'] = offsets['MEDIUM'] + field_data.get('MEDIUM_TO_HARD_SPEED', 0.5)

    for c in ['SOFT', 'MEDIUM', 'HARD']:
        models[c] = AnchoredModel(quali_pace, deg_slopes[c], offsets[c])
    return models

def get_race_result_grid(year, event_name):
    """
    FALLBACK: Gets the actual GridPosition from the final Race Result ('R') 
    if the Qualifying ('Q') data is unavailable.
    """
    try:
        print(f"   🏁 FALLBACK: Fetching Race Grid for {event_name}...")
        session = fastf1.get_session(year, event_name, 'R')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        
        results = session.results
        grid_map = {}
        
        for driver_code in results['Abbreviation']:
            try:
                row = results[results['Abbreviation'] == driver_code]
                pos = row['GridPosition'].values[0]
                
                # If 0.0 (Pit Lane/DQ), assign them to back of grid (P20)
                if pos <= 0.0: pos = 20.0
                
                grid_map[driver_code] = int(pos)
            except:
                continue
                
        if len(grid_map) > 0:
            pole = min(grid_map, key=grid_map.get)
            print(f"      ✅ Race Grid Loaded! Pole was {pole}")
            return grid_map
        else:
            return {}
            
    except Exception as e:
        print(f"      ⚠️ Race Grid Fetch Failed: {e}")
        return {}


def get_actual_grid(year, event_name):
    """
    PRIMARY GRID FETCH: Tries Qualifying ('Q'), then falls back to Race Result ('R').
    """
    grid = {}
    
    # 1. Try Qualifying ('Q')
    try:
        print(f"   🏁 Fetching Official Qualifying Grid for {event_name}...")
        session = fastf1.get_session(year, event_name, 'Q')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results
        
        for driver_code in results['Abbreviation']:
            try:
                row = results[results['Abbreviation'] == driver_code]
                pos = row['GridPosition'].values[0]
                if pos <= 0.0: pos = 20.0
                grid[driver_code] = int(pos)
            except: continue
            
        if len(grid) > 0:
            pole = min(grid, key=grid.get)
            print(f"      ✅ Q Grid Loaded! Pole: {pole}")
            return grid
        
    except Exception as e:
        # If Q fails, fall through to the next check
        print(f"      ⚠️ Q Grid Fetch Failed: {e}")

    # 2. Try Race Result ('R')
    if not grid:
        grid = get_race_result_grid(year, event_name)

    return grid


def get_driver_recent_form(driver, year, current_round, lookback=5):
    """
    ENHANCED: Scans the last 'lookback' races with weighted recency.
    More recent races get higher weight. Returns form index and confidence score.
    """
    form_scores = []
    weights = []
    rounds_analyzed = []
    
    start_round = max(1, current_round - lookback)
    
    for i, r in enumerate(range(start_round, current_round)):
        try:
            session = fastf1.get_session(year, r, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            
            clean = session.laps.pick_quicklaps()
            if clean.empty: continue
            
            # Field median pace
            field_median = clean['LapTime'].dt.total_seconds().median()
            
            # Driver pace
            driver_laps = clean[clean['Driver'] == driver]
            if driver_laps.empty: continue
            
            driver_median = driver_laps['LapTime'].dt.total_seconds().median()
            score = driver_median / field_median
            
            # Weight more recent races higher (exponential)
            weight = np.exp(i / lookback)
            
            form_scores.append(score)
            weights.append(weight)
            rounds_analyzed.append(session.event['EventName'])
            
        except Exception:
            continue
            
    if not form_scores:
        return 1.0, [], 0.0  # No data
    
    # Weighted average
    weighted_form = np.average(form_scores, weights=weights)
    
    # Confidence: based on number of races and consistency
    confidence = len(form_scores) / lookback * (1.0 - np.std(form_scores))
    confidence = np.clip(confidence, 0.0, 1.0)
        
    return weighted_form, rounds_analyzed, confidence

def load_prediction_data(year, round_num, event_name):
    """
    FUSION ENGINE V6:
    1. Loads Practice Data (Pace).
    2. Normalizes Pace (Track Evolution Correction).
    3. Fetches Robust Grid.
    """
    valid_laps_list = []
    loaded_sources = []
    max_total_laps = 57 
    session_medians = {}
    
    sessions_to_try = ['FP1', 'FP2', 'FP3']
    print(f"   📡 Scanning {year} sessions for {event_name}...")
    
    # 1. LOAD PACE DATA
    raw_sessions = {}
    for s_name in sessions_to_try:
        try:
            session = fastf1.get_session(year, round_num, s_name)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            
            if len(session.laps) > 50:
                clean = session.laps.pick_quicklaps()
                if not clean.empty:
                    median_pace = clean['LapTime'].dt.total_seconds().median()
                    session_medians[s_name] = median_pace
                    
                    laps = session.laps
                    laps['SourceSession'] = s_name
                    raw_sessions[s_name] = laps
                    
                    if hasattr(session, 'total_laps') and session.total_laps:
                        max_total_laps = session.total_laps
        except: pass

    # Fallback to historical data if no practice data found
    if not raw_sessions:
        try:
            print("      ⚠️ No Practice Data. Loading Historical Race Data...")
            session = fastf1.get_session(year - 1, round_num, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            session.official_grid = {} 
            return session, f"Race Data ({year-1}) [Historical]"
        except: return None, "No Data"

    # 2. NORMALIZE
    fastest_session = min(session_medians, key=session_medians.get)
    target_pace = session_medians[fastest_session]
    final_laps = []
    for s_name, laps in raw_sessions.items():
        offset = session_medians[s_name] - target_pace
        offset_td = pd.to_timedelta(offset, unit='s')
        laps['LapTime'] = laps['LapTime'] - offset_td
        final_laps.append(laps)
        loaded_sources.append(s_name)

    merged_laps = pd.concat(final_laps, ignore_index=True)
    
    # 3. GET OFFICIAL GRID (Robust Fetch)
    clean_name = event_name.replace(" Grand Prix", "").strip()
    real_grid = get_actual_grid(year, clean_name)
    
    final_session = CombinedSession(merged_laps, max_total_laps, event_name, year)
    final_session.official_grid = real_grid
    
    return final_session, " + ".join(loaded_sources)

def filter_practice_long_runs(laps, min_stint_length=4):
    """
    ENHANCED: Finds consistent stints with fuel-load awareness.
    Filters for race-simulation runs (longer stints, heavier fuel).
    """
    if not isinstance(laps, pd.DataFrame) or laps.empty: return pd.DataFrame()
    laps = laps.copy()
    
    # Group by Session AND Stint
    if 'SourceSession' in laps.columns:
        laps['StintLaps'] = laps.groupby(['Driver', 'SourceSession', 'Stint'])['LapNumber'].transform('count')
    else:
        laps['StintLaps'] = laps.groupby(['Driver', 'Stint'])['LapNumber'].transform('count')
    
    # Filter for meaningful race-sim stints
    long_runs = laps[laps['StintLaps'] >= min_stint_length]
    
    # Additional quality filter: remove obvious outliers within each stint
    if not long_runs.empty and 'LapTime' in long_runs.columns:
        # For each stint, filter laps within 107% of median
        def filter_stint_outliers(stint_df):
            if len(stint_df) < 3:
                return stint_df
            median_time = stint_df['LapTime'].dt.total_seconds().median()
            max_time = median_time * 1.07  # 107% rule
            valid = stint_df['LapTime'].dt.total_seconds() <= max_time
            return stint_df[valid]
        
        if 'SourceSession' in long_runs.columns:
            long_runs = long_runs.groupby(['Driver', 'SourceSession', 'Stint'], group_keys=False).apply(
                filter_stint_outliers,
                include_groups=False
            )
        else:
            long_runs = long_runs.groupby(['Driver', 'Stint'], group_keys=False).apply(
                filter_stint_outliers,
                include_groups=False
            )
        
    return long_runs