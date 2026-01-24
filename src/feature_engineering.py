"""
Feature Engineering Module for F1 Race Prediction
Extracts rich features from historical race data with minimal fixed data
"""

import fastf1
import pandas as pd
import numpy as np
from collections import defaultdict


def extract_driver_historical_features(driver_code, year, current_round, lookback=5):
    """
    Extract comprehensive features from last N races for a driver.
    Returns dict with multiple performance indicators.
    """
    features = {
        'avg_finish_position': 20.0,
        'avg_quali_position': 20.0,
        'avg_positions_gained': 0.0,
        'consistency_score': 0.0,
        'dnf_rate': 0.0,
        'reliability_risk': 0.0,  
        'avg_pace_vs_teammate': 1.0,
        'tire_management_score': 1.0,  
        'qualifying_conversion': 1.0, 
        'overtaking_ability': 0.0,  
        'race_craft': 0.0,  
    }
    
    finish_positions = []
    quali_positions = []
    positions_gained = []
    pace_ratios = []
    tire_management = []
    early_gains = []
    late_gains = []
    dnf_count = 0
    races_analyzed = 0
    
    start_round = max(1, current_round - lookback)
    
    for round_num in range(start_round, current_round):
        try:
            # Load race session
            race_session = fastf1.get_session(year, round_num, 'R')
            race_session.load(laps=False, telemetry=False, weather=False, messages=False)
            
            # Get driver's result
            driver_result = race_session.results[race_session.results['Abbreviation'] == driver_code]
            if driver_result.empty:
                continue
                
            driver_result = driver_result.iloc[0]
            
            # Finish position
            try:
                finish_pos = int(driver_result['Position'])
                if finish_pos > 0:
                    finish_positions.append(finish_pos)
            except:
                dnf_count += 1
                continue
            
            # Quali position
            try:
                quali_pos = int(driver_result['GridPosition'])
                if quali_pos > 0:
                    quali_positions.append(quali_pos)
                    positions_gained.append(quali_pos - finish_pos)
            except:
                pass
            
            # Load lap data for deeper analysis
            try:
                race_session.load(laps=True, telemetry=False, weather=False, messages=False)
                driver_laps = race_session.laps[race_session.laps['Driver'] == driver_code]
                
                if not driver_laps.empty:
                    # Pace vs teammate
                    team = driver_result['TeamName']
                    teammate_laps = race_session.laps[
                        (race_session.laps['Team'] == team) & 
                        (race_session.laps['Driver'] != driver_code)
                    ]
                    
                    if not teammate_laps.empty:
                        driver_median = driver_laps['LapTime'].dt.total_seconds().median()
                        teammate_median = teammate_laps['LapTime'].dt.total_seconds().median()
                        if teammate_median > 0:
                            pace_ratios.append(driver_median / teammate_median)
                    
                    # Tire management (degradation consistency)
                    for stint in driver_laps['Stint'].unique():
                        stint_laps = driver_laps[driver_laps['Stint'] == stint]
                        if len(stint_laps) >= 5:
                            times = stint_laps['LapTime'].dt.total_seconds().values
                            # Lower std dev = better tire management
                            tire_management.append(np.std(times))
                    
                    # Early vs late race performance (positions gained)
                    if quali_pos > 0:
                        # Position at lap 10
                        lap_10 = driver_laps[driver_laps['LapNumber'] == 10]
                        if not lap_10.empty:
                            pos_lap_10 = lap_10.iloc[0]['Position']
                            early_gains.append(quali_pos - pos_lap_10)
                            late_gains.append(pos_lap_10 - finish_pos)
            except:
                pass
            
            races_analyzed += 1
            
        except Exception as e:
            continue
    
    # Calculate aggregate features
    if races_analyzed > 0:
        if finish_positions:
            features['avg_finish_position'] = np.mean(finish_positions)
            features['consistency_score'] = np.std(finish_positions)
        
        if quali_positions:
            features['avg_quali_position'] = np.mean(quali_positions)
            
        if positions_gained:
            features['avg_positions_gained'] = np.mean(positions_gained)
            
        if quali_positions and finish_positions:
            features['qualifying_conversion'] = np.mean(quali_positions) / np.mean(finish_positions)
        
        features['dnf_rate'] = dnf_count / (races_analyzed + dnf_count)
        
        if pace_ratios:
            features['avg_pace_vs_teammate'] = np.mean(pace_ratios)
        
        if tire_management:
            # Normalize: lower is better
            features['tire_management_score'] = np.mean(tire_management) / 2.0  # Typical std is ~2s
        
        if early_gains:
            features['overtaking_ability'] = np.mean(early_gains)
        
        if late_gains:
            features['race_craft'] = np.mean(late_gains)
    
    return features, races_analyzed


def extract_track_characteristics(year, race_name, lookback_years=2):
    """
    Learn track-specific patterns from historical data at this circuit.
    Returns dict with track characteristics.
    """
    characteristics = {
        'overtaking_difficulty': 0.5,  # 0=easy, 1=hard
        'tire_deg_severity': 1.0,  # Multiplier for degradation
        'safety_car_likelihood': 0.15,
        'avg_positions_changed': 5.0,
        'compound_preference': 'MEDIUM',  # Most successful compound
        'qualifying_importance': 0.7,  # How much quali matters (correlation)
    }
    
    # Track-specific overrides for known problem circuits
    overrides = {
        'Qatar': {'qualifying_importance': 0.65, 'overtaking_difficulty': 0.7, 'tire_deg_severity': 1.3},
        'Las Vegas': {'qualifying_importance': 0.75, 'overtaking_difficulty': 0.8, 'safety_car_likelihood': 0.25},
        'Netherlands': {'qualifying_importance': 0.8, 'overtaking_difficulty': 0.85, 'tire_deg_severity': 0.9},
        'Monaco': {'qualifying_importance': 0.95, 'overtaking_difficulty': 0.95, 'safety_car_likelihood': 0.35},
        'Singapore': {'qualifying_importance': 0.85, 'overtaking_difficulty': 0.75, 'safety_car_likelihood': 0.30},
    }
    
    # Apply override if track matches
    for track_key, override_values in overrides.items():
        if track_key in race_name:
            characteristics.update(override_values)
            print(f"      🎯 Applied override for {track_key}")
            return characteristics
    
    try:
        # Analyze recent years at this track
        positions_changed_list = []
        quali_correlations = []
        deg_indicators = []
        sc_count = 0
        races_found = 0
        compound_performance = defaultdict(list)
        pit_losses = []
        
        for year_offset in range(lookback_years + 1):
            check_year = year - year_offset
            if check_year < 2020:  # Don't go too far back
                continue
                
            try:
                session = fastf1.get_session(check_year, race_name, 'R')
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                
                results = session.results
                
                # Overtaking difficulty (how many positions change)
                positions_changed = abs(results['GridPosition'] - results['Position']).sum()
                positions_changed_list.append(positions_changed)
                
                # Qualifying importance (correlation between grid and finish)
                valid_results = results[(results['GridPosition'] > 0) & (results['Position'] > 0)]
                if len(valid_results) > 5:
                    corr = valid_results['GridPosition'].corr(valid_results['Position'])
                    quali_correlations.append(corr)
                
                # Load laps for deeper analysis
                try:
                    session.load(laps=True, telemetry=False, weather=False, messages=False)
                    
                    # Safety car detection (laps with unusually slow times)
                    all_laps = session.laps.pick_quicklaps()
                    if not all_laps.empty:
                        median_lap = all_laps['LapTime'].dt.total_seconds().median()
                        slow_laps = session.laps[
                            session.laps['LapTime'].dt.total_seconds() > median_lap * 1.3
                        ]
                        if len(slow_laps) > 10:  # Likely SC period
                            sc_count += 1
                    
                    # Tire degradation severity (analyze stint length variance)
                    stint_lengths = session.laps.groupby(['Driver', 'Stint'])['LapNumber'].count()
                    if not stint_lengths.empty:
                        deg_indicators.append(stint_lengths.std())
                    
                    # Compound performance
                    for compound in ['SOFT', 'MEDIUM', 'HARD']:
                        compound_laps = all_laps[all_laps['Compound'] == compound]
                        if len(compound_laps) > 10:
                            avg_time = compound_laps['LapTime'].dt.total_seconds().mean()
                            compound_performance[compound].append(avg_time)
                            
                    # Pit Loss Calculation (In-Lap + Out-Lap Delta)
                    pit_in_laps = session.laps[session.laps['PitInTime'].notna()]
                    pit_out_laps = session.laps[session.laps['PitOutTime'].notna()]
                    
                    if not pit_in_laps.empty and not pit_out_laps.empty:
                        # Median clean lap
                        median_pace = all_laps['LapTime'].dt.total_seconds().median()
                        
                        # Calculate In-Lap Loss (Entry)
                        in_lap_times = pit_in_laps['LapTime'].dt.total_seconds()
                        # Filter crazy outliers (e.g., changing nose)
                        in_lap_times = in_lap_times[in_lap_times < median_pace + 30] 
                        if not in_lap_times.empty:
                            avg_in_loss = in_lap_times.median() - median_pace
                        else: avg_in_loss = 3.0 # Fallback
                        
                        # Calculate Out-Lap Loss (Exit + Stationary)
                        out_lap_times = pit_out_laps['LapTime'].dt.total_seconds()
                        out_lap_times = out_lap_times[out_lap_times < median_pace + 40]
                        if not out_lap_times.empty:
                            avg_out_loss = out_lap_times.median() - median_pace
                        else: avg_out_loss = 19.0 # Fallback
                        
                        total_loss = avg_in_loss + avg_out_loss
                        if 15.0 < total_loss < 35.0: # Sanity check for F1 pit stops
                             pit_losses.append(total_loss)
                
                except:
                    pass
                
                races_found += 1
                
            except:
                continue
        
        # Aggregate characteristics
        if races_found > 0:
            if positions_changed_list:
                avg_changes = np.mean(positions_changed_list)
                characteristics['avg_positions_changed'] = avg_changes
                # More changes = easier overtaking
                characteristics['overtaking_difficulty'] = max(0.1, 1.0 - (avg_changes / 100))
            
            if quali_correlations:
                characteristics['qualifying_importance'] = np.mean(quali_correlations)
            
            characteristics['safety_car_likelihood'] = sc_count / max(races_found, 1)
            
            if deg_indicators:
                # Higher variance = more severe degradation
                avg_deg = np.mean(deg_indicators)
                characteristics['tire_deg_severity'] = 0.5 + (avg_deg / 10.0)
            
            # Find best compound
            if compound_performance:
                best_compound = min(compound_performance.keys(), 
                                  key=lambda c: np.mean(compound_performance[c]))
                characteristics['compound_preference'] = best_compound

            # PIT LOSS CALCULATION
            if pit_losses:
                 characteristics['avg_pit_loss'] = np.mean(pit_losses)
                 print(f"      Calculated Pit Loss: {characteristics['avg_pit_loss']:.1f}s")
            else:
                 characteristics['avg_pit_loss'] = 22.0

            
    except Exception as e:
        print(f"   ⚠️ Track learning limited: {e}")
    
    return characteristics


def extract_driver_compound_affinity(driver_code, year, current_round, lookback=5):
    """
    Learn how well a driver performs on different compounds relative to the field.
    Returns dict: {compound: relative_performance_multiplier}
    """
    affinity = {
        'SOFT': 1.0,
        'MEDIUM': 1.0,
        'HARD': 1.0
    }
    
    compound_times = {c: [] for c in ['SOFT', 'MEDIUM', 'HARD']}
    field_compound_times = {c: [] for c in ['SOFT', 'MEDIUM', 'HARD']}
    
    start_round = max(1, current_round - lookback)
    
    for round_num in range(start_round, current_round):
        try:
            session = fastf1.get_session(year, round_num, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            
            driver_laps = session.laps[session.laps['Driver'] == driver_code].pick_quicklaps()
            all_laps = session.laps.pick_quicklaps()
            
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                # Driver's pace on compound
                driver_compound = driver_laps[driver_laps['Compound'] == compound]
                if not driver_compound.empty:
                    driver_median = driver_compound['LapTime'].dt.total_seconds().median()
                    compound_times[compound].append(driver_median)
                
                # Field average on compound
                field_compound = all_laps[all_laps['Compound'] == compound]
                if not field_compound.empty:
                    field_median = field_compound['LapTime'].dt.total_seconds().median()
                    field_compound_times[compound].append(field_median)
        
        except:
            continue
    
    # Calculate relative performance
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        if compound_times[compound] and field_compound_times[compound]:
            driver_avg = np.mean(compound_times[compound])
            field_avg = np.mean(field_compound_times[compound])
            
            if field_avg > 0:
                # < 1.0 means driver is better than field on this compound
                affinity[compound] = driver_avg / field_avg
    
    return affinity


def build_feature_matrix(year, race_name, current_round, driver_codes):
    """
    Build complete feature matrix for all drivers.
    Returns DataFrame with all features.
    """
    print(f"\n   🧠 Building Enhanced Feature Matrix...")
    
    # Get track characteristics once
    track_chars = extract_track_characteristics(year, race_name)
    print(f"      Track Overtaking Difficulty: {track_chars['overtaking_difficulty']:.2f}")
    print(f"      Qualifying Importance: {track_chars['qualifying_importance']:.2f}")
    
    feature_rows = []
    
    for i, driver in enumerate(driver_codes):
        print(f"\r      Analyzing driver {i+1}/{len(driver_codes)}: {driver}", end="")
        
        # Historical performance features
        hist_features, races_used = extract_driver_historical_features(driver, year, current_round)
        
        # Compound affinity
        compound_affinity = extract_driver_compound_affinity(driver, year, current_round)
        
        # Combine all features
        row = {
            'Driver': driver,
            'races_analyzed': races_used,
            **hist_features,
            **{f'{comp}_affinity': compound_affinity[comp] for comp in ['SOFT', 'MEDIUM', 'HARD']},
            **{f'track_{k}': v for k, v in track_chars.items()}
        }
        
        feature_rows.append(row)
    
    print("\n      ✅ Feature matrix complete")
    return pd.DataFrame(feature_rows), track_chars
