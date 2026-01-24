import numpy as np
import random
from src.reliability_model import apply_reliability_penalty

def calculate_sc_time(pit_loss_avg):
    return 8.0 

def simulate_race_history(models, strategy_list, total_laps, grid_position=1, consistencies=0.12, 
                         driver_form=1.0, safety_car_prob=0.10, driver_code="", session_context=None,
                         driver_features=None, track_characteristics=None, compound_affinity=None,
                         reliability_mode: str = "probabilistic", ml_correction=0.0):
    """
    ENHANCED STABILITY ENGINE V5.0
    Uses rich feature set for realistic race simulation with track-specific behavior.
    
    Args:
        models: Tire performance models
        strategy_list: [(compound, laps), ...]
        total_laps: Race distance
        grid_position: Starting position
        consistencies: Driver consistency (std dev)
        driver_form: Recent form multiplier
        safety_car_prob: SC probability
        driver_code: Driver abbreviation
        session_context: Quali pace reference
        driver_features: Dict of historical features
        track_characteristics: Dict of track-specific data
        compound_affinity: Dict of driver's compound preferences
    """
    history = []
    current_time = 0.0
    laps_completed = 0
    
    # === ENHANCED FEATURE INTEGRATION ===
    
    # 1. Base tier modifier from quali pace (scaled by qualifying importance)
    if session_context and driver_code in session_context:
        all_times = list(session_context.values())
        best_in_session = min(all_times)
        driver_time = session_context[driver_code]
        performance_gap = driver_time / best_in_session
        # Qualifying importance from track characteristics (0..1)
        qual_imp = 0.7
        if track_characteristics:
            qual_imp = float(track_characteristics.get('qualifying_importance', 0.7))
        # Reduced weighting: 0.3 + 0.3 instead of 0.5 + 0.4 for softer impact
        alpha = np.clip(0.3 + 0.3 * qual_imp, 0.3, 0.6)
        tier_modifier = alpha * performance_gap + (1.0 - alpha)
    else:
        tier_modifier = 1.05
    
    # 2. Enhanced driver behavior from features
    WHISPERERS = ['ALB', 'PER', 'SAI', 'BOT', 'HUL', 'MAG']
    CHARGERS = ['VER', 'NOR', 'LEC', 'HAM', 'PIA', 'RUS', 'ALO']
    
    is_whisperer = driver_code in WHISPERERS
    is_charger = driver_code in CHARGERS
    
    # Tire management from features
    if driver_features and 'tire_management_score' in driver_features:
        # Lower score = better management = less degradation
        deg_mult = 0.85 + (driver_features['tire_management_score'] * 0.15)
    elif is_whisperer:
        deg_mult = 0.92
    else:
        deg_mult = 1.0
    
    # Overtaking ability from features
    if driver_features and 'overtaking_ability' in driver_features:
        # More positions gained historically = better in traffic
        overtake_power = 0.8 + (driver_features['overtaking_ability'] * 0.05)
        overtake_power = np.clip(overtake_power, 0.3, 1.5)
    elif is_charger:
        overtake_power = 0.5
    else:
        overtake_power = 1.0
    
    # Race craft (late-race performance)
    race_craft_bonus = 0.0
    if driver_features and 'race_craft' in driver_features:
        race_craft_bonus = driver_features['race_craft'] * 0.02  # Small bonus per position gained
    
    # 3. Track-specific modifiers
    if track_characteristics:
        traffic_severity = track_characteristics.get('overtaking_difficulty', 0.5)
        deg_severity = track_characteristics.get('tire_deg_severity', 1.0)
        track_evolution_rate = 0.015 if track_characteristics.get('qualifying_importance', 0.7) > 0.75 else 0.010
        reliability_stress = float(track_characteristics.get('reliability_stress', 1.0))
    else:
        traffic_severity = 0.5
        deg_severity = 1.0
        track_evolution_rate = 0.010
        reliability_stress = 1.0
    
    # 4. Universal physics
    FUEL_PENALTY = 0.033
    TRACK_EVO = track_evolution_rate
    sigma = consistencies
    
    
    # 5. Compound affinity multipliers
    compound_multipliers = {}
    
    # Get Dynamic Pit Loss
    PIT_LOSS = 22.0
    if track_characteristics:
         PIT_LOSS = track_characteristics.get('avg_pit_loss', 22.0)
         
    if compound_affinity:
        for comp in ['SOFT', 'MEDIUM', 'HARD']:
            compound_multipliers[comp] = compound_affinity.get(f'{comp}_affinity', 1.0)
    else:
        compound_multipliers = {'SOFT': 1.0, 'MEDIUM': 1.0, 'HARD': 1.0}
    
    # 6. ML Correction (Bias Application)
    # Distribute the correction across laps to maintain smooth graph
    # If ML says driver is 10s slower (positive), we add time to each lap.
    lap_bias = ml_correction / max(1, total_laps)
    
    # === RACE SIMULATION ===
    for stint_idx, (compound, laps_duration) in enumerate(strategy_list):
        model = models[compound]
        stint_len_planned = int(laps_duration)
        compound_mult = compound_multipliers.get(compound, 1.0)
        
        for lap_in_stint in range(1, total_laps - laps_completed + 1):
            laps_completed += 1
            
            is_sc = (random.random() < (safety_car_prob / total_laps))
            must_pit = (lap_in_stint == stint_len_planned) and laps_completed < total_laps
            
            if is_sc:
                lap_time = 92.0 + random.gauss(0, 0.5)
            else:
                # Base tire model pace
                tire_age = float(lap_in_stint * deg_mult * deg_severity)
                base_pace = model.predict(tire_age)
                
                # Apply modifiers
                effective_pace = base_pace * tier_modifier * compound_mult
                
                # Traffic (reduces over time, affected by track characteristics)
                traffic_adj = (grid_position - 1) * 0.04 * traffic_severity * overtake_power * np.exp(-0.1 * laps_completed)
                
                # Fuel effect (decreasing weight)
                fuel_effect = (110.0 - (laps_completed * (105.0 / total_laps))) * FUEL_PENALTY
                
                # Track evolution (lap times drop as track rubbers in)
                track_evo_effect = -laps_completed * TRACK_EVO
                
                # Late-race craft bonus (kicks in after 50% distance)
                race_progress = laps_completed / total_laps
                craft_effect = race_craft_bonus if race_progress > 0.5 else 0.0
                
                # Driver form and consistency
                form_effect = effective_pace * (driver_form - 1.0)
                consistency_noise = random.gauss(0, sigma)
                
                # Tire Warmup Penalty (Cold tires on out-lap)
                warmup_penalty = 0.0
                if lap_in_stint == 1 and laps_completed > 1: # Out-lap (not race start)
                     if compound == 'HARD': warmup_penalty = 3.5
                     elif compound == 'MEDIUM': warmup_penalty = 1.5
                     elif compound == 'SOFT': warmup_penalty = 0.5
                
                     elif compound == 'SOFT': warmup_penalty = 0.5
                
                # Re-entry Traffic (Dirty Air after Pit Stop)
                # Simulates getting stuck behind slower cars if pitting early (Dense field)
                traffic_penalty = 0.0
                if lap_in_stint <= 3 and laps_completed > 5: # First 3 laps of a post-pit stint
                     # Field density is high early in the race (Laps 1-20)
                     # Penalty decays as race progresses (field spreads out)
                     field_spread_factor = max(0.0, 1.0 - (laps_completed / (total_laps * 0.7)))
                     # Higher grid position (slower car) = more likely to effect you? 
                     # Actually, if you are fast (Grid 1) and pit early, you fall into traffic (Grid 10-20).
                     # So penalty is HIGHER for fast cars pitting early.
                     if grid_position <= 8: # Top teams suffer most from traffic
                         traffic_penalty = random.uniform(0.5, 1.5) * field_spread_factor * traffic_severity
                     else:
                         traffic_penalty = random.uniform(0.0, 0.5) * field_spread_factor * traffic_severity

                lap_time = (effective_pace + form_effect + fuel_effect + 
                           track_evo_effect + traffic_adj + craft_effect + consistency_noise + lap_bias + warmup_penalty + traffic_penalty)
            
            # First lap adjustment
            if laps_completed == 1:
                lap_time += 4.0
            
            current_time += float(lap_time)
            history.append(current_time)
            
            # Pit decision
            if must_pit or (is_sc and 10 < laps_completed < total_laps - 10):
                current_time += PIT_LOSS if not is_sc else 8.0
                break
            
            if laps_completed == total_laps:
                # Apply reliability penalty at finish (DNF modeling / conservative risk)
                if driver_features and ('reliability_risk' in driver_features):
                    final_time = history[-1]
                    adjusted_time, dnf = apply_reliability_penalty(
                        final_time,
                        driver_features,
                        track_factor=reliability_stress,
                        sim_mode=reliability_mode,
                    )
                    history[-1] = adjusted_time
                    return history, ("DNF" if dnf else "Finished")
                return history, "Finished"
    
    # Apply reliability if we somehow exit loop without explicit finish
    if driver_features and ('reliability_risk' in driver_features) and history:
        final_time = history[-1]
        adjusted_time, dnf = apply_reliability_penalty(
            final_time,
            driver_features,
            track_factor=reliability_stress,
            sim_mode=reliability_mode,
        )
        history[-1] = adjusted_time
        return history, ("DNF" if dnf else "Finished")
    return history, "Finished"