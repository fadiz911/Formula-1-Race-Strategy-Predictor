"""
Batch evaluator: runs the full prediction pipeline across multiple (year, race)
cases and reports correlation and podium accuracy per race.

Usage examples:
  # Single case
  python batch_evaluate.py --case "2024,Abu Dhabi Grand Prix"

  # Multiple cases
  python batch_evaluate.py \
    --case "2024,Abu Dhabi Grand Prix" \
    --case "2024,United States Grand Prix" \
    --case "2024,Qatar Grand Prix"

  # Control runtime
  python batch_evaluate.py --sim-count 100 --max-drivers 14 --case "2024,Abu Dhabi Grand Prix"
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
import fastf1

from typing import List, Tuple

from src.predictor import load_prediction_data, filter_practice_long_runs, get_driver_recent_form
from src.strategy_engine import train_tire_model, calculate_field_deltas
from src.optimizer import optimize_strategy
from src.simulation import simulate_race_history
from src.evaluator import evaluate_sim_performance
from src.feature_engineering import (
    extract_driver_historical_features,
    extract_track_characteristics,
    extract_driver_compound_affinity,
)


def get_round_from_name(year: int, race_name: str) -> int:
    try:
        schedule = fastf1.get_event_schedule(year)
        key = race_name.replace(" Grand Prix", "").strip()
        race = schedule[schedule['EventName'].str.contains(key, case=False, na=False)]
        if not race.empty:
            return int(race.iloc[0]['RoundNumber'])
    except Exception:
        pass
    fallback = {
        "Bahrain": 1, "Saudi": 2, "Australia": 3, "Japan": 4, "China": 5, "Miami": 6,
        "Emilia": 7, "Monaco": 8, "Canada": 9, "Spain": 10, "Austria": 11, "Britain": 12,
        "Hungary": 13, "Belgium": 14, "Netherlands": 15, "Italy": 16, "Azerbaijan": 17,
        "Singapore": 18, "United": 19, "Mexico": 20, "Brazil": 21, "Las": 22,
        "Qatar": 23, "Abu": 24
    }
    for k, v in fallback.items():
        if k in race_name:
            return v
    raise ValueError(f"Could not resolve round number for {year} {race_name}")


class AnchoredModel:
    def __init__(self, quali_pace: float, deg_slope: float, compound_offset: float):
        self.base_pace = float(quali_pace) + float(compound_offset)
        self.deg_slope = float(deg_slope)

    def predict(self, X):
        X = np.array(X)
        if len(X.shape) > 1:
            lap_age = X[:, 0]
        else:
            lap_age = X.flatten()
        prediction = self.base_pace + (lap_age * self.deg_slope)
        return float(prediction.item()) if prediction.size == 1 else prediction.astype(float)


def train_anchored_models(session, driver: str, fastest_laps: pd.Series):
    try:
        quali_pace = float(fastest_laps[driver])
    except Exception:
        quali_pace = float(np.median(fastest_laps.values))

    deg_slopes = {'SOFT': 0.12, 'MEDIUM': 0.08, 'HARD': 0.05}

    try:
        # Use available practice/race laps to tune slopes per compound
        all_laps = session.laps
        laps = all_laps[all_laps['Driver'] == driver]
        laps = laps.dropna(subset=['LapTime'])
        laps = laps[laps['PitInTime'].isna() & laps['PitOutTime'].isna()]
        laps = filter_practice_long_runs(laps)
        if not laps.empty:
            for comp in ['SOFT', 'MEDIUM', 'HARD']:
                model = train_tire_model(laps, comp)
                if model:
                    t1 = model.predict([[1]])
                    t2 = model.predict([[2]])
                    slope = max(0.0, float(t2) - float(t1))
                    if slope > 0.25:
                        slope = 0.15
                    deg_slopes[comp] = slope
    except Exception:
        pass

    offsets = {'SOFT': 0.0, 'MEDIUM': 0.5, 'HARD': 1.0}
    models = {c: AnchoredModel(quali_pace, deg_slopes[c], offsets[c]) for c in ['SOFT', 'MEDIUM', 'HARD']}
    return models


def parse_cases(args: argparse.Namespace) -> List[Tuple[int, str]]:
    cases = []
    for entry in args.case:
        try:
            year_str, race_name = entry.split(',', 1)
            cases.append((int(year_str.strip()), race_name.strip()))
        except ValueError:
            print(f"Ignoring invalid --case entry: '{entry}'. Expected 'YEAR,RACE NAME'.")
    return cases


def run_case(year: int, race_name: str, sim_count: int, max_drivers: int, consistencies: float) -> dict:
    round_num = get_round_from_name(year, race_name)

    # Load data
    session, src = load_prediction_data(year, round_num, race_name)
    df = session.laps.dropna(subset=['LapTime']).copy()
    df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
    fastest_laps = df.groupby('Driver')['LapTimeSec'].min().sort_values()

    # Grid
    if hasattr(session, 'official_grid') and session.official_grid:
        grid_drivers = sorted(session.official_grid.keys(), key=lambda x: session.official_grid[x])
        grid_drivers = [d for d in grid_drivers if d in fastest_laps.index]
    else:
        grid_drivers = fastest_laps.head(22).index.tolist()

    grid_drivers = grid_drivers[:max_drivers]
    total_laps = int(getattr(session, 'total_laps', 57))

    # Contexts
    track_chars = extract_track_characteristics(year, race_name)
    session_context = fastest_laps.to_dict()

    results_agg = []

    for i, driver in enumerate(grid_drivers):
        models = train_anchored_models(session, driver, fastest_laps)
        strat_name, strat_laps, _ = optimize_strategy(models, total_laps)

        if "1-Stop" in strat_name:
            strat_list = [('SOFT', int(strat_laps)), ('HARD', int(total_laps - int(strat_laps)))]
        else:
            l1, l2 = strat_laps
            strat_list = [('SOFT', int(l1)), ('MEDIUM', int(l2 - l1)), ('HARD', int(total_laps - l2))]

        # Start position
        start_pos = i + 1
        if hasattr(session, 'official_grid') and driver in session.official_grid:
            start_pos = int(session.official_grid[driver])

        # Features
        form_index, _, _ = get_driver_recent_form(driver, year, round_num)
        driver_features, _ = extract_driver_historical_features(driver, year, round_num)
        compound_affinity = extract_driver_compound_affinity(driver, year, round_num)

        times = []
        for _ in range(sim_count):
            hist, status = simulate_race_history(
                models, strat_list, total_laps,
                grid_position=start_pos,
                consistencies=consistencies,
                driver_form=form_index,
                safety_car_prob=track_chars.get('safety_car_likelihood', 0.15),
                driver_code=driver,
                session_context=session_context,
                driver_features=driver_features,
                track_characteristics=track_chars,
                compound_affinity=compound_affinity,
            )
            times.append(hist[-1])

        avg_t = float(np.mean(times)) if times else 99999.0
        results_agg.append({
            'Driver': driver,
            'AvgTime': avg_t,
            'Range': float(np.ptp(times)) if times else 0.0,
            'Strategy': strat_name,
            'Strat_laps': strat_laps,
            'StartPos': start_pos,
            'RawTimes': times,
        })

    results_agg = sorted(results_agg, key=lambda x: x['AvgTime'])

    audit = evaluate_sim_performance(year, race_name, results_agg)
    return {
        'year': year,
        'race': race_name,
        'correlation': float(audit['correlation']) if audit else None,
        'podium_hits': int(audit['podium_accuracy']) if audit else None,
        'count': len(results_agg),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate correlations across races.")
    parser.add_argument('--case', action='append', required=True, help="'YEAR,RACE NAME' (repeatable)")
    parser.add_argument('--sim-count', type=int, default=120, help='Simulations per driver (default: 120)')
    parser.add_argument('--max-drivers', type=int, default=16, help='Max drivers to simulate (default: 16)')
    parser.add_argument('--consistency', type=float, default=0.12, help='Driver consistency sigma (default: 0.12)')
    args = parser.parse_args()

    cases = parse_cases(args)
    print("\n============================================")
    print(" BATCH CORRELATION EVALUATION ")
    print("============================================\n")

    results = []
    start = time.time()
    for idx, (year, race) in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] Running {year} — {race} ...")
        try:
            res = run_case(year, race, args.sim_count, args.max_drivers, args.consistency)
            print(f"   → Spearman: {res['correlation']:.2f} | Podium: {res['podium_hits']}/3 | Drivers: {res['count']}")
            results.append(res)
        except Exception as e:
            print(f"   ⚠️ Case failed: {e}")
            results.append({'year': year, 'race': race, 'correlation': None, 'podium_hits': None, 'count': 0})
        print()

    df = pd.DataFrame(results)
    print("\nSUMMARY:\n")
    print(df.to_markdown(index=False))

    if df['correlation'].notna().any():
        avg_corr = df['correlation'].dropna().mean()
        print(f"\nAverage Spearman across {df['correlation'].notna().sum()} cases: {avg_corr:.2f}")

    print(f"\nTotal runtime: {time.time()-start:.1f}s")


if __name__ == '__main__':
    main()
