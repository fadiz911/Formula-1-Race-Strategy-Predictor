import numpy as np
import random

def calculate_sc_time(pit_loss_avg):
    """
    Returns the effective pit stop time under Safety Car/VSC.
    Under SC, pit loss is drastically reduced as the field slows down.
    """
    # Time lost in the pit lane itself (travel + stop) under slow conditions
    SC_PIT_LOSS = 8.0 
    return SC_PIT_LOSS 

def simulate_race_history(models, strategy_list, total_laps, grid_position=1, consistencies=None, pit_loss_avg=22.0, driver_form=1.0, safety_car_prob=0.20):
    """
    Runs a SINGLE probabilistic simulation with REACTIONARY PHYSICS.
    Includes: Form, Fuel, Track Evo, Traffic, Safety Car/Reactive Pit, and type hardening.
    """
    history = []
    current_time = 0.0
    
    # --- PHYSICS CONSTANTS ---
    START_FUEL = 110.0
    BURN_RATE = 105.0 / total_laps 
    FUEL_PENALTY = 0.035
    TRACK_EVO = 0.04
    
    # Reactive Tire Management Constants
    CRITICAL_DEGRADATION_THRESHOLD = 0.15 # s/lap/lap increase
    PIT_LOSS_UNDER_SC = 8.0 
    
    current_fuel = START_FUEL
    laps_completed = 0
    sigma = consistencies if consistencies is not None else 0.2
    
    current_tire_deg_slope = 0.0
    
    for stint_idx, (compound, laps_duration) in enumerate(strategy_list):
        model = models[compound]
        stint_len_planned = int(laps_duration)
        
        # We need to manually control the loop to allow for early exits (cliffs/SC)
        # We loop up to the remaining laps in the race
        for lap_in_stint in range(1, total_laps - laps_completed + 1):
            
            # --- 1. PRE-LAP CHECKS ---
            
            laps_completed += 1
            
            # 1a. Safety Car Check (Random Event per lap)
            # The probability is spread across all laps (e.g., 20% over 50 laps = 0.4% chance per lap)
            is_safety_car = (random.random() < (safety_car_prob / total_laps))
            
            # 1b. Reactive Pit Check (Tire Cliff)
            # We only check for a tire cliff if we are past the halfway mark of the planned stint.
            is_tire_cliff = (current_tire_deg_slope > CRITICAL_DEGRADATION_THRESHOLD and 
                             lap_in_stint > (stint_len_planned / 2) and
                             laps_completed < total_laps)
            
            # Decide if pit is needed this lap
            is_last_lap_of_stint_planned = (lap_in_stint == stint_len_planned)
            
            must_pit_this_lap = False
            pit_time_spent = 0.0
            
            # Override 1: SC Check (Always pit if SC comes out near ideal pit window)
            if is_safety_car and laps_completed > 5 and laps_completed < total_laps - 10:
                must_pit_this_lap = True
                pit_time_spent = calculate_sc_time(pit_loss_avg) 
            
            # Override 2: Tire Cliff Check (Emergency pit)
            elif is_tire_cliff and not is_last_lap_of_stint_planned:
                must_pit_this_lap = True
                pit_time_spent = pit_loss_avg # Normal green flag pit loss
            
            # Normal planned pit stop
            elif is_last_lap_of_stint_planned and laps_completed < total_laps:
                must_pit_this_lap = True
                pit_time_spent = pit_loss_avg # Normal green flag pit loss


            # --- 2. CALCULATE PACE ---
            
            lap_time = 0.0 # Initialize as float
            
            if is_safety_car:
                lap_time = 80.0 + random.gauss(0, 1.0) # SC Laps are ~80 seconds
            else:
                # Pace based on Model * Form
                # FIX: model.predict now returns a scalar float (no [0])
                base_time = model.predict([[lap_in_stint]]) * driver_form
                
                # Physics
                fuel_cost = current_fuel * FUEL_PENALTY
                track_gain = laps_completed * TRACK_EVO
                
                # Traffic/Dirty Air
                traffic_lag = 0.0
                if grid_position > 1:
                    clearance_luck = random.uniform(0.8, 1.2)
                    traffic_lag = (grid_position * 0.08 * clearance_luck) * np.exp(-0.25 * laps_completed)
                
                # Noise
                noise = random.gauss(0, sigma)
                
                lap_time = base_time + fuel_cost - track_gain + traffic_lag + noise
            
            # Standing Start Penalty
            if laps_completed == 1: lap_time += 4.5
            
            # --- 3. PHYSICS UPDATE ---
            # FIX: Ensure lap_time is a float before accumulation
            current_time += float(lap_time)
            current_fuel -= BURN_RATE
            
            history.append(current_time)
            
            # --- 4. POST-LAP CHECKS (PIT & DEGRADATION) ---

            # Estimate Degradation Slope for Reactive Check
            if laps_completed > 1 and not is_safety_car:
                 # Check the change in pace relative to the last lap
                 current_deg_estimate = lap_time - history[-2] 
                 # We subtract the speed gain expected from fuel burn/track evolution (~0.075s/lap total)
                 current_tire_deg_slope = max(0.0, current_deg_estimate - 0.075)

            # Pit Stop Execution
            if must_pit_this_lap and laps_completed < total_laps:
                # FIX: Ensure pit_time_spent is a float before accumulation
                current_time += float(pit_time_spent)
                
                # We need to break out of this inner loop to move to the next stint
                break 

            # Fuel DNF Check
            if current_fuel <= 0: return history, f"DNF (Fuel) L{laps_completed}"

            # If the race finishes this lap
            if laps_completed == total_laps:
                return history, "Finished"

    # If the simulation somehow finishes the outer stint loop before total_laps
    if laps_completed < total_laps:
        return history, f"DNF (Strategy Error) L{laps_completed}"
        
    return history, "Finished"