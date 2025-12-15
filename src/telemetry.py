import fastf1
import pandas as pd

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