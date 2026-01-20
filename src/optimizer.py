from src.simulation import simulate_race_history

def optimize_strategy(models, total_laps):
    best_time = 1e9 
    best_strat_name = "1-Stop"
    best_strat_laps = int(total_laps / 2)
    best_hist = []

    # Search windows
    start_window = int(total_laps * 0.15)
    end_window = int(total_laps * 0.75)
    
    for pit_lap in range(start_window, end_window):
        strategy = [('SOFT', pit_lap), ('HARD', total_laps - pit_lap)]
        # Pass dummy values for adaptive features to prevent errors
        hist, _ = simulate_race_history(
            models, strategy, total_laps, 
            consistencies=0.0, 
            driver_code="", 
            session_context=None
        )
        
        try:
            race_time = float(hist[-1])
            if race_time < best_time:
                best_time, best_strat_name, best_strat_laps, best_hist = race_time, "1-Stop (S-H)", pit_lap, hist
        except: continue

    # Repeat for 2-Stop...
    p1_start, p1_end = int(total_laps * 0.10), int(total_laps * 0.35)
    for pit1 in range(p1_start, p1_end): 
        for pit2 in range(pit1 + 10, int(total_laps * 0.80)):
            strategy = [('SOFT', pit1), ('MEDIUM', pit2 - pit1), ('HARD', total_laps - pit2)]
            hist, _ = simulate_race_history(models, strategy, total_laps, consistencies=0.0, driver_code="", session_context=None)
            try:
                race_time = float(hist[-1])
                if race_time < best_time:
                    best_time, best_strat_name, best_strat_laps, best_hist = race_time, "2-Stop (S-M-H)", (pit1, pit2), hist
            except: continue

    return best_strat_name, best_strat_laps, best_hist