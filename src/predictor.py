import fastf1
import pandas as pd
import numpy as np
import os

# --- 1. ENABLE CACHE ---
# Create 'cache' folder if it doesn't exist and enable fastf1 caching
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

class CombinedSession:
    """
    A 'Virtual Session' object to hold merged practice data and the official grid map.
    """
    def __init__(self, laps, total_laps, event_name, year):
        self.laps = laps
        self.total_laps = total_laps
        self.event = {'EventName': event_name}
        self.official_grid = {} 

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
    Scans the last 'lookback' races (same season) to calculate driver form.
    Form Index < 1.0 means they are driving faster than the field average.
    """
    form_scores = []
    rounds_analyzed = []
    
    start_round = max(1, current_round - lookback)
    
    for r in range(start_round, current_round):
        try:
            # Load only Lap Data (Lightweight)
            session = fastf1.get_session(year, r, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            
            # Normalize pace by dividing driver median by field median
            clean = session.laps.pick_quicklaps()
            if clean.empty: continue
            
            field_median = clean['LapTime'].dt.total_seconds().median()
            driver_laps = clean[clean['Driver'] == driver]
            if driver_laps.empty: continue
            
            driver_median = driver_laps['LapTime'].dt.total_seconds().median()
            score = driver_median / field_median
            
            form_scores.append(score)
            rounds_analyzed.append(session.event['EventName'])
            
        except Exception:
            continue
            
    if not form_scores:
        return 1.0, [] # No data? Assume average form.
        
    return np.mean(form_scores), rounds_analyzed

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

def filter_practice_long_runs(laps):
    """
    Finds consistent stints of 4+ laps for reliable degradation modeling.
    """
    if not isinstance(laps, pd.DataFrame) or laps.empty: return pd.DataFrame()
    laps = laps.copy()
    
    # Group by Session AND Stint
    if 'SourceSession' in laps.columns:
        laps['StintLaps'] = laps.groupby(['Driver', 'SourceSession', 'Stint'])['LapNumber'].transform('count')
    else:
        laps['StintLaps'] = laps.groupby(['Driver', 'Stint'])['LapNumber'].transform('count')
        
    return laps[laps['StintLaps'] >= 4]