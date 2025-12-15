from src.simulation import simulate_race_history

def optimize_strategy(models, total_laps):
    """
    Simulates thousands of race combinations to find the theoretical fastest strategy.
    Runs simulations with zero noise (consistencies=0.0) for deterministic comparison.
    """
    # FIX: Initialize best_time as a large, known float to prevent TypeErrors
    best_time = 1000000.0 
    best_strat_name = "1-Stop"
    best_strat_laps = int(total_laps / 2)
    best_hist = []

    # --- 1. EVALUATE 1-STOP ---
    start_window = int(total_laps * 0.2)
    end_window = int(total_laps * 0.8)
    
    for pit_lap in range(start_window, end_window):
        # We test Soft -> Hard strategy first
        strategy = [('SOFT', pit_lap), ('HARD', total_laps - pit_lap)]
        
        # Simulate with zero noise (consistencies=0.0) and ignore status output
        hist, _ = simulate_race_history(models, strategy, total_laps, consistencies=0.0)
        
        # FIX: Explicitly ensure race_time is a float before comparison
        try:
            race_time = float(hist[-1])
        except (IndexError, TypeError, ValueError):
            # If hist is empty or corrupted, assign a heavy DNF penalty
            race_time = 999999.0 
            
        if race_time < best_time:
            best_time = race_time
            best_strat_name = "1-Stop"
            best_strat_laps = pit_lap
            best_hist = hist

    # --- 2. EVALUATE 2-STOP ---
    start_p1 = int(total_laps * 0.15)
    end_p1 = int(total_laps * 0.40)
    
    for pit1 in range(start_p1, end_p1, 2):
        start_p2 = pit1 + 15
        end_p2 = int(total_laps * 0.85)
        
        for pit2 in range(start_p2, end_p2, 2):
            # We test Soft -> Hard -> Soft strategy
            strategy = [('SOFT', pit1), ('HARD', pit2 - pit1), ('SOFT', total_laps - pit2)]
            
            hist, _ = simulate_race_history(models, strategy, total_laps, consistencies=0.0)
            
            # FIX: Explicitly ensure race_time is a float before comparison
            try:
                race_time = float(hist[-1])
            except (IndexError, TypeError, ValueError):
                race_time = 999999.0 
            
            if race_time < best_time:
                best_time = race_time
                best_strat_name = "2-Stop"
                best_strat_laps = (pit1, pit2)
                best_hist = hist

    return best_strat_name, best_strat_laps, best_hist