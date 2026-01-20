"""
Season evaluator for a given year (default 2025).
- Lists all drivers per race
- Runs prediction per race
- Outputs misplacement (Predicted vs Actual) per driver
- Produces per-race CSVs and a season aggregate CSV

Usage:
  source .venv/bin/activate
  python season_evaluate.py --year 2025 --sim-count 120 --max-drivers 16

Options:
  --year: season year (default: 2025)
  --sim-count: simulations per driver (default: 120)
  --max-drivers: cap number of drivers per race for speed (default: 16; set 20 for full grid)
  --consistency: driver consistency sigma (default: 0.12)
  --out-dir: output directory (default: reports/season_<year>)
  --round-start/--round-end: limit rounds (1-indexed)
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import fastf1

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
from src.reliability_model import (
    extract_reliability_profile,
    get_track_stress_multiplier,
    apply_reliability_penalty,
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


def train_anchored_models(session, driver: str, fastest_laps: pd.Series, track_chars: dict = None, driver_features: dict = None):
    try:
        quali_pace = float(fastest_laps[driver])
    except Exception:
        quali_pace = float(np.median(fastest_laps.values))

    deg_slopes = {'SOFT': 0.12, 'MEDIUM': 0.08, 'HARD': 0.05}
    # Apply track tire degradation severity and driver tire management influence
    track_deg = 1.0
    if track_chars:
        try:
            track_deg = float(track_chars.get('tire_deg_severity', 1.0))
        except Exception:
            track_deg = 1.0
    driver_deg_mult = 1.0
    if driver_features and ('tire_management_score' in driver_features):
        try:
            score = float(driver_features.get('tire_management_score', 1.0))
            # Lower score (better management) should reduce degradation multiplier
            driver_deg_mult = np.clip(0.85 + 0.25 * score, 0.85, 1.35)
        except Exception:
            driver_deg_mult = 1.0

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

    # Final adjustment to degradation slopes
    for c in deg_slopes:
        deg_slopes[c] = float(deg_slopes[c]) * track_deg * driver_deg_mult

    offsets = {'SOFT': 0.0, 'MEDIUM': 0.5, 'HARD': 1.0}
    models = {c: AnchoredModel(quali_pace, deg_slopes[c], offsets[c]) for c in ['SOFT', 'MEDIUM', 'HARD']}
    return models


def ensure_dirs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'drivers'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'evaluations'), exist_ok=True)


def get_season_events(year: int, round_start: int = 1, round_end: int = None) -> List[Tuple[int, str]]:
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule.sort_values('RoundNumber')
    if round_end is None:
        round_end = int(schedule['RoundNumber'].max())
    events = []
    for _, row in schedule.iterrows():
        rnd = int(row['RoundNumber'])
        if round_start <= rnd <= round_end:
            events.append((rnd, str(row['EventName'])))
    return events


def run_race(year: int, event_name: str, sim_count: int, max_drivers: int, consistency: float, use_reliability: bool = True):
    round_num = get_round_from_name(year, event_name)
    session, src = load_prediction_data(year, round_num, event_name)

    # Driver list (actual)
    try:
        race_session = fastf1.get_session(year, event_name, 'R')
        race_session.load(laps=False, telemetry=False, weather=False, messages=False)
        actual_drivers = race_session.results['Abbreviation'].tolist()
    except Exception:
        # Fallback to practice laps
        actual_drivers = sorted(session.laps['Driver'].dropna().unique().tolist())

    # Pace context
    df = session.laps.dropna(subset=['LapTime']).copy()
    df['LapTimeSec'] = df['LapTime'].dt.total_seconds()
    fastest_laps = df.groupby('Driver')['LapTimeSec'].min().sort_values()

    # Grid for simulation
    if hasattr(session, 'official_grid') and session.official_grid:
        grid_drivers = sorted(session.official_grid.keys(), key=lambda x: session.official_grid[x])
        grid_drivers = [d for d in grid_drivers if d in fastest_laps.index]
    else:
        grid_drivers = fastest_laps.head(22).index.tolist()

    grid_drivers = grid_drivers[:max_drivers]
    total_laps = int(getattr(session, 'total_laps', 57))

    # Contexts
    track_chars = extract_track_characteristics(year, event_name)
    session_context = fastest_laps.to_dict()
    
    # Reliability context
    track_reliability_stress = get_track_stress_multiplier(event_name) if use_reliability else 1.0

    results_agg = []

    for i, driver in enumerate(grid_drivers):
        # Features
        form_index, _, _ = get_driver_recent_form(driver, year, round_num)
        driver_features, _ = extract_driver_historical_features(driver, year, round_num)
        compound_affinity = extract_driver_compound_affinity(driver, year, round_num)
        # Reliability profile
        if use_reliability:
            reliability_profile = extract_reliability_profile(driver, year, round_num, lookback=10)
            driver_features['reliability_risk'] = reliability_profile['reliability_risk'] * track_reliability_stress

        models = train_anchored_models(session, driver, fastest_laps, track_chars=track_chars, driver_features=driver_features)
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

        times = []
        dnf_count = 0
        for _ in range(sim_count):
            hist, status = simulate_race_history(
                models, strat_list, total_laps,
                grid_position=start_pos,
                consistencies=consistency,
                driver_form=form_index,
                safety_car_prob=track_chars.get('safety_car_likelihood', 0.15),
                driver_code=driver,
                session_context=session_context,
                driver_features=driver_features,
                track_characteristics=track_chars,
                compound_affinity=compound_affinity,
                reliability_mode='probabilistic' if use_reliability else 'conservative',
            )
            final_time = hist[-1]
            times.append(final_time)
            if status == 'DNF':
                dnf_count += 1

        # If DNF rate is high in sims, heavily penalize predicted position
        avg_t = float(np.mean(times)) if times else 99999.0
        if dnf_count > sim_count * 0.3:  # >30% DNF rate in sims
            avg_t *= 1.2  # Push down in predictions
        # Grid bias: on tracks where qualifying matters and overtaking is hard,
        # penalize deeper grid positions with a small time add to stabilize ranking
        try:
            qual_imp = float(track_chars.get('qualifying_importance', 0.7))
            traffic_severity = float(track_chars.get('overtaking_difficulty', 0.5))
            # Reduced from 0.4+1.0 to 0.15+0.35 - less aggressive
            base_bias = 0.15 + 0.35 * qual_imp * traffic_severity  # seconds per grid place
            grid_bias = (start_pos - 1) * base_bias
            avg_t += grid_bias
        except Exception:
            pass
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

    # Evaluation (with misplacement)
    audit = evaluate_sim_performance(year, event_name, results_agg)
    df_eval = None
    corr = None
    podium = None
    if audit:
        df_eval = audit['data'].copy()
        df_eval['Misplacement'] = df_eval['Predicted'] - df_eval['Actual']
        df_eval['AbsMisplacement'] = df_eval['Misplacement'].abs()
        corr = float(audit['correlation'])
        podium = int(audit['podium_accuracy'])

    return {
        'results_agg': results_agg,
        'drivers_actual': actual_drivers,
        'evaluation': df_eval,
        'correlation': corr,
        'podium': podium,
        'total_laps': total_laps,
    }


def main():
    parser = argparse.ArgumentParser(description="Run season evaluation and output misplacements per race.")
    parser.add_argument('--year', type=int, default=2025, help='Season year (default: 2025)')
    parser.add_argument('--sim-count', type=int, default=120, help='Simulations per driver (default: 120)')
    parser.add_argument('--max-drivers', type=int, default=16, help='Max drivers to simulate (default: 16; use 20 for full grid)')
    parser.add_argument('--consistency', type=float, default=0.12, help='Driver consistency sigma (default: 0.12)')
    parser.add_argument('--out-dir', default=None, help='Output directory (default: reports/season_<year>)')
    parser.add_argument('--round-start', type=int, default=1, help='Start round (default: 1)')
    parser.add_argument('--round-end', type=int, default=None, help='End round (default: last)')
    parser.add_argument('--use-reliability', action='store_true', default=True, help='Enable reliability/DNF modeling (default: True)')
    parser.add_argument('--no-reliability', action='store_false', dest='use_reliability', help='Disable reliability modeling')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join('reports', f'season_{args.year}')
    ensure_dirs(out_dir)

    # Enumerate season events
    events = get_season_events(args.year, args.round_start, args.round_end)
    print("\n============================================")
    print(f" SEASON EVALUATION {args.year} — Rounds {args.round_start}..{events[-1][0] if events else '?'} ")
    print("============================================\n")

    season_rows = []
    summary_rows = []

    t0 = time.time()
    for idx, (rnd, event_name) in enumerate(events, start=1):
        print(f"[{idx}/{len(events)}] {args.year} — R{rnd} — {event_name}")
        try:
            res = run_race(args.year, event_name, args.sim_count, args.max_drivers, args.consistency, args.use_reliability)

            # Save drivers list
            drivers_path = os.path.join(out_dir, 'drivers', f"{args.year:04d}_R{rnd:02d}_{event_name.replace(' ', '_')}_drivers.txt")
            with open(drivers_path, 'w') as f:
                for d in res['drivers_actual']:
                    f.write(str(d) + "\n")

            # Save evaluation with misplacement
            if res['evaluation'] is not None:
                df_eval = res['evaluation'].copy()
                df_eval['Year'] = args.year
                df_eval['Round'] = rnd
                df_eval['Race'] = event_name
                season_rows.append(df_eval)

                eval_path = os.path.join(out_dir, 'evaluations', f"{args.year:04d}_R{rnd:02d}_{event_name.replace(' ', '_')}_evaluation.csv")
                df_eval[['Year','Round','Race','Driver','Start','Actual','Predicted','Misplacement','AbsMisplacement']].to_csv(eval_path, index=False)

            summary_rows.append({
                'Year': args.year,
                'Round': rnd,
                'Race': event_name,
                'Correlation': res['correlation'],
                'PodiumHits': res['podium'],
                'DriversSimulated': len(res['results_agg']) if res['results_agg'] else 0,
                'TotalLaps': res['total_laps'],
            })
            print(f"   → Corr: {res['correlation'] if res['correlation'] is not None else 'NA'} | Podium: {res['podium'] if res['podium'] is not None else 'NA'} | Drivers: {len(res['results_agg'])}")
        except Exception as e:
            print(f"   ⏭️ Skipped: {e}")
        print()

    # Aggregate season output
    if season_rows:
        df_season = pd.concat(season_rows, ignore_index=True)
        agg_path = os.path.join(out_dir, f"{args.year}_season_misplacements.csv")
        df_season[['Year','Round','Race','Driver','Start','Actual','Predicted','Misplacement','AbsMisplacement']].to_csv(agg_path, index=False)

    df_summary = pd.DataFrame(summary_rows)
    sum_path = os.path.join(out_dir, f"{args.year}_season_summary.csv")
    df_summary.to_csv(sum_path, index=False)

    print("SUMMARY:\n")
    try:
        print(df_summary.to_markdown(index=False))
    except Exception:
        print(df_summary)

    if df_summary['Correlation'].notna().any():
        avg_corr = df_summary['Correlation'].dropna().mean()
        print(f"\nAverage Spearman: {avg_corr:.2f}")

    print(f"\nSeason evaluation written to: {out_dir}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
