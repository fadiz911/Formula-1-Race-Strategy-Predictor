"""
Reliability and DNF prediction model
Addresses the #1 issue: Model doesn't predict retirements (avg 10.5 positions off)
"""

import fastf1
import numpy as np
import pandas as pd
from collections import defaultdict


def extract_reliability_profile(driver_code, year, current_round, lookback=10):
    """
    Build reliability risk profile for a driver based on recent DNF history.
    
    Returns:
        dict with:
        - dnf_rate: overall DNF rate
        - recent_dnf_rate: exponentially weighted recent rate
        - reliability_risk: 0-1 probability of DNF this race
        - confidence: how much historical data we have
    """
    dnf_count = 0
    finish_count = 0
    recent_dnfs = []  # Weighted by recency
    
    start_round = max(1, current_round - lookback)
    
    for i, round_num in enumerate(range(start_round, current_round)):
        try:
            session = fastf1.get_session(year, round_num, 'R')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            
            driver_result = session.results[session.results['Abbreviation'] == driver_code]
            if driver_result.empty:
                continue
            
            driver_result = driver_result.iloc[0]
            status = str(driver_result.get('Status', ''))
            position = driver_result.get('Position', 0)
            
            # Check if DNF/retirement
            is_dnf = False
            if pd.notna(position) and position > 0:
                finish_count += 1
            else:
                is_dnf = True
                dnf_count += 1
            
            # Could also check: 'DNF' in status or 'Retired' in status
            if '+' not in str(position) and position == 0:
                is_dnf = True
            
            # Weight recent races more (exponential decay)
            weight = np.exp(i / lookback)
            recent_dnfs.append(1.0 if is_dnf else 0.0)
            recent_dnfs[-1] *= weight
            
        except Exception:
            continue
    
    total_races = dnf_count + finish_count
    
    # Overall DNF rate
    dnf_rate = dnf_count / max(1, total_races)
    
    # Recent weighted DNF rate
    if recent_dnfs:
        recent_rate = sum(recent_dnfs) / max(1, len(recent_dnfs))
    else:
        recent_rate = 0.0
    
    # Reliability risk: blend of recent and overall, capped
    # More weight to recent DNFs
    reliability_risk = min(0.35, 0.7 * recent_rate + 0.3 * dnf_rate)
    
    # Confidence based on data availability
    confidence = min(1.0, total_races / lookback)
    
    return {
        'dnf_rate': float(dnf_rate),
        'recent_dnf_rate': float(recent_rate),
        'reliability_risk': float(reliability_risk),
        'confidence': float(confidence),
        'races_analyzed': int(total_races),
    }


def get_track_reliability_factor(year, race_name, lookback_years=3):
    """
    Some tracks are historically harder on reliability (heat, high-speed, street circuits).
    
    Returns:
        float: reliability stress factor (1.0 = normal, >1.0 = higher DNF risk)
    """
    dnf_rates = []
    
    for year_offset in range(lookback_years):
        check_year = year - year_offset
        if check_year < 2020:
            continue
        
        try:
            session = fastf1.get_session(check_year, race_name, 'R')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            
            results = session.results
            # Count DNFs (position = NaN or 0)
            dnf_count = len(results[results['Position'].isna() | (results['Position'] == 0)])
            total_starters = len(results)
            
            if total_starters > 0:
                dnf_rates.append(dnf_count / total_starters)
        except Exception:
            continue
    
    if dnf_rates:
        avg_dnf_rate = np.mean(dnf_rates)
        # Normalize: typical F1 DNF rate is ~10-15%
        # Above 20% = high stress track
        return 1.0 + max(0, (avg_dnf_rate - 0.15) * 2.0)
    
    return 1.0


def apply_reliability_penalty(predicted_time, driver_features, track_factor=1.0, sim_mode='probabilistic'):
    """
    Apply reliability risk to prediction.
    
    Args:
        predicted_time: base predicted race time
        driver_features: dict with 'reliability_risk' key
        track_factor: track-specific reliability stress
        sim_mode: 'probabilistic' (random DNF) or 'conservative' (time penalty)
    
    Returns:
        adjusted_time: with reliability factored in
        dnf_occurred: bool, True if simulated DNF
    """
    reliability_risk = driver_features.get('reliability_risk', 0.0) * track_factor
    
    if sim_mode == 'probabilistic':
        # Randomly trigger DNF based on risk
        if np.random.random() < reliability_risk:
            # DNF: return very high time (last place equivalent)
            return predicted_time * 1.5, True
        else:
            return predicted_time, False
    
    elif sim_mode == 'conservative':
        # Apply time penalty proportional to risk (no random DNF)
        # Higher risk = slower predicted time (conservative estimate)
        penalty = 1.0 + (reliability_risk * 0.1)  # Up to 10% time penalty
        return predicted_time * penalty, False
    
    return predicted_time, False


# Track categorization for reliability stress
TRACK_CATEGORIES = {
    # High stress (heat, altitude, high-speed)
    'Bahrain': 1.3,
    'Saudi Arabia': 1.2,
    'Qatar': 1.2,  # Reduced from 1.4 - was too harsh
    'Singapore': 1.3,
    'Mexico': 1.2,
    'Brazil': 1.2,
    'Austria': 1.2,
    'Monza': 1.2,
    
    # Street circuits (crash risk)
    'Monaco': 1.4,
    'Baku': 1.3,
    'Las Vegas': 1.2,  # Reduced from 1.3
    
    # Normal stress
    'Silverstone': 1.0,
    'Spa': 1.0,
    'Suzuka': 1.0,
}


def get_track_stress_multiplier(race_name):
    """
    Quick lookup for track-specific reliability stress.
    """
    for key, mult in TRACK_CATEGORIES.items():
        if key in race_name:
            return mult
    return 1.0  # Default
