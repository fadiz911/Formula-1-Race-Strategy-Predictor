
import fastf1
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_driver_laps(session, driver):
    """
    Robust lap getter that works with both FastF1 Sessions and our 'CombinedSession'.
    """
    # 1. Get the dataframe safely
    if hasattr(session, 'laps'):
        laps = session.laps
    else:
        laps = session # It is already a dataframe
        
    # 2. Filter by Driver
    driver_laps = laps[laps['Driver'] == driver].copy()
    
    if driver_laps.empty:
        return driver_laps
        
    return driver_laps

def get_telemetry_features(session, driver):
    """
    Extracts aggregated telemetry features for a driver from a session (typically Qualifying).
    Returns a dictionary of features.
    """
    try:
        # Get all laps for the driver
        laps = session.laps.pick_drivers(driver)
        
        # Pick the fastest lap for representative "peak" performance
        # Or alternatively, pick all valid quick laps and average.
        # Let's use the fastest lap to see their 'ceiling' potential.
        fastest_lap = laps.pick_fastest()
        
        if fastest_lap is None or pd.isna(fastest_lap['LapTime']):
            return {}

        # Load telemetry
        # fastf1 might warn if telemetry is not loaded, but session.load() usually handles it
        try:
            tel = fastest_lap.get_telemetry()
        except Exception:
            # Telemetry might not be available
            return {}
            
        if tel.empty:
            return {}
            
        # Features calculation
        # Speed
        avg_speed = tel['Speed'].mean()
        max_speed = tel['Speed'].max()
        
        # Throttle
        # Avg throttle application
        avg_throttle = tel['Throttle'].mean()
        # Full throttle percentage (frames where throttle > 99)
        full_throttle_pct = (tel['Throttle'] > 99).mean() * 100
        
        # Brake
        # Avg brake application (boolean or pressure depending on car data, usually boolean 0/100 or pressure)
        avg_brake = tel['Brake'].mean()
        
        # Gear
        # Avg gear (lower might mean more torque/cornering focus?)
        avg_gear = tel['nGear'].mean()
        # Gear changes count
        gear_changes = (tel['nGear'].diff().abs() > 0).sum()

        return {
            'Qualy_AvgSpeed': avg_speed,
            'Qualy_MaxSpeed': max_speed,
            'Qualy_AvgThrottle': avg_throttle,
            'Qualy_FullThrottlePct': full_throttle_pct,
            'Qualy_AvgBrake': avg_brake,
            'Qualy_AvgGear': avg_gear,
            'Qualy_GearChanges': gear_changes
        }
        
    except Exception as e:
        logger.warning(f"Error getting telemetry for {driver}: {e}")
        return {}

def get_actual_strategy(session, driver):
    """
    Extracts the actual tire strategy used by a driver in the session.
    Merges consecutive stints on the same compound to avoid 'ghost' stops.
    Returns a string like: "S (18) -> M (20) -> H (20)"
    """
    try:
        if not hasattr(session, 'laps'): return "N/A"
        
        laps = session.laps
        
        # 1. CLEANING: Remove duplicate laps and sort
        d_laps = laps[laps['Driver'] == driver].sort_values('LapNumber').drop_duplicates(subset=['LapNumber'])
        
        if d_laps.empty: return "DNS"
        
        # 2. FILLING: Propagate compound forward/backward (Crucial for fixing flickering M->S->M)
        # Often data has gaps. We assume compound doesn't change unless explicit.
        d_laps['Compound'] = d_laps['Compound'].replace('', pd.NA).ffill().bfill()
        
        stints = []
        current_stint = {'compound': None, 'laps': 0}
        
        for idx, row in d_laps.iterrows():
            comp = row['Compound']
            if pd.isna(comp): comp = "UNK"
            
            # Start first stint
            if current_stint['compound'] is None:
                current_stint['compound'] = comp
                current_stint['laps'] = 1
                continue
            
            # Check for change
            if comp != current_stint['compound']:
                stints.append(current_stint)
                current_stint = {'compound': comp, 'laps': 1}
            else:
                current_stint['laps'] += 1
                
        if current_stint['compound']:
            stints.append(current_stint)
            
        # Format
        comp_map = {'SOFT': 'S', 'MEDIUM': 'M', 'HARD': 'H', 'INTERMEDIATE': 'I', 'WET': 'W', 'UNK': '?'}
        
        summary_parts = []
        for s in stints:
            c_name = str(s['compound']).upper()
            c_code = comp_map.get(c_name, c_name[0] if c_name else '?')
            summary_parts.append(f"{c_code} ({s['laps']})")
            
        return " -> ".join(summary_parts)
        
    except Exception as e:
        return f"Error: {e}"
    
    # 3. Filter "Quick Laps" Manually (Equivalent to pick_quicklaps)
    # - Must have a valid LapTime
    # - Must NOT be an In-Lap (PitInTime is NaT)
    # - Must NOT be an Out-Lap (PitOutTime is NaT)
    
    # Drop laps with no time
    driver_laps = driver_laps.dropna(subset=['LapTime'])
    
    # Drop In/Out laps
    driver_laps = driver_laps[driver_laps['PitInTime'].isna() & driver_laps['PitOutTime'].isna()]
    
    # Optional: Filter out super slow laps (> 115% of median)
    if len(driver_laps) > 5:
        median_time = driver_laps['LapTime'].dt.total_seconds().median()
        driver_laps = driver_laps[driver_laps['LapTime'].dt.total_seconds() < (median_time * 1.15)]
    
    return driver_laps