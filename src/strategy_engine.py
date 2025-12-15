import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def train_tire_model(laps_data, compound='SOFT'):
    # (Keep your existing train_tire_model code here)
    # Just ensure it handles empty inputs gracefully
    if laps_data.empty: return None
    # ... rest of the function ...
    # (Use the code from previous steps, checking for slope clamping)
    # ...
    # Placeholder return for brevity in this snippet, 
    # MAKE SURE TO KEEP YOUR FULL IMPLEMENTATION
    return None 

def calculate_field_deltas(session):
    """
    Robust field analysis for merged datasets.
    """
    print("   🕵️‍♂️ Analyzing Field Physics (Merged Data)...")
    
    speed_diffs_sm = []
    deg_diffs_sm = []
    speed_diffs_mh = []
    deg_diffs_mh = []
    all_medium_paces = []

    test_drivers = ['VER', 'NOR', 'LEC', 'SAI', 'RUS', 'HAM', 'PIA', 'ALO', 'GAS', 'TSU']
    
    # Safe access to laps
    all_laps = session.laps
    
    for driver in test_drivers:
        # Manual Filter (Same as telemetry.py)
        laps = all_laps[all_laps['Driver'] == driver].copy()
        laps = laps.dropna(subset=['LapTime'])
        laps = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()]
        
        if laps.empty: continue
            
        driver_stats = {}
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            c_laps = laps[laps['Compound'] == compound]
            
            # We need at least 3 laps to calculate a slope
            if len(c_laps) >= 3:
                X = c_laps[['TyreLife']].values
                y = c_laps['LapTime'].dt.total_seconds().values
                
                if len(y) > 1:
                    reg = LinearRegression().fit(X, y)
                    slope = reg.coef_[0]
                else:
                    slope = 0.1
                
                # Safety Clamp
                if slope < -0.05: slope = 0.0
                if slope > 0.25: slope = 0.15
                    
                median_pace = np.median(y)
                driver_stats[compound] = (median_pace, slope)
                
                if compound == 'MEDIUM':
                    all_medium_paces.append(median_pace)
        
        # Calculate Deltas
        if 'SOFT' in driver_stats and 'MEDIUM' in driver_stats:
            speed_diffs_sm.append(driver_stats['MEDIUM'][0] - driver_stats['SOFT'][0])
            deg_diffs_sm.append(driver_stats['SOFT'][1] - driver_stats['MEDIUM'][1])

        if 'MEDIUM' in driver_stats and 'HARD' in driver_stats:
            speed_diffs_mh.append(driver_stats['HARD'][0] - driver_stats['MEDIUM'][0])
            deg_diffs_mh.append(driver_stats['MEDIUM'][1] - driver_stats['HARD'][1])

    # Averages
    deltas = {}
    deltas['SOFT_TO_MEDIUM_SPEED'] = np.mean(speed_diffs_sm) if speed_diffs_sm else 0.8
    deltas['SOFT_TO_MEDIUM_DEG'] = np.mean(deg_diffs_sm) if deg_diffs_sm else 0.08
    deltas['MEDIUM_TO_HARD_SPEED'] = np.mean(speed_diffs_mh) if speed_diffs_mh else 0.6
    deltas['MEDIUM_TO_HARD_DEG'] = np.mean(deg_diffs_mh) if deg_diffs_mh else 0.05
    
    # Fail-safe average pace
    if all_medium_paces:
        deltas['AVERAGE_PACE'] = np.mean(all_medium_paces)
    else:
        deltas['AVERAGE_PACE'] = 90.0
    
    print(f"      > Field Physics: Softs are {deltas['SOFT_TO_MEDIUM_SPEED']:.2f}s faster")
    return deltas