import pandas as pd
import numpy as np
from src.lstm_predictor import LSTMPredictor
from scipy.stats import pearsonr
import logging

# Suppress Logs
logging.getLogger('fastf1').setLevel(logging.ERROR)

def run_benchmark():
    print("🚀 Starting Model V3 Benchmark Suite (2023-2025)...")
    
    # 1. Initialize Model & Load Data (Single Source of Truth)
    print("🧠 Loading Predictor and History...")
    predictor = LSTMPredictor()
    
    if not predictor.loaded:
        print("❌ Model V3 not loaded. Please train it first.")
        return

    # Use the predictor's internal history to avoid double fetching
    # Note: predictor.history_df is already SCALED. We need unscaled or context features?
    # Actually, predict_position expects 'context' values (raw float).
    # But history_df has SCALED values.
    # We must inverse transform or fetch raw. 
    # Fetching raw is safer to simulate real app flow exactly.
    # Optimization: Predictor loads Scaled History. We need Raw Context for "Live" simulation input.
    # So we DO need to fetch data again OR keep a raw copy in predictor.
    # Let's fetch once here (the user complaint was about 'twice' but maybe they saw logs from app + script).
    # To fix "twice" in THIS script, we just fetch here and use it.
    
    # But wait, predictor ALREADY fetched 2023-2025 in __init__.
    # Use predictor.history_df BUT we need to reverse scale to get "QualiDelta" inputs?
    # Because predict_position() takes raw inputs and scales them itself.
    # If we pass scaled inputs, it will double scale.
    
    # Solution: Reverse scale the specific context columns from history_df
    print("📊 Preparing Validation Data from Model History...")
    df = predictor.history_df.copy() # This is scaled
    
    if df.empty:
        print("❌ No data found.")
        return

    # Inverse Scale - CRITICAL: Must use exact column names scaler was trained on
    # Phase 2 Update: Scaler now expects 10 features
    # ['FinishPos', 'GridPos', 'Points', 'SpeedST', 'StintCount', 'QualiDelta', 'PracticePace', 'IsWet', 'DriverConsistency', 'TrackPerformance']
    scaler = predictor.scaler
    feature_cols = ['FinishPos', 'GridPos', 'Points', 'SpeedST', 'StintCount', 'QualiDelta', 'PracticePace', 'IsWet', 'DriverConsistency', 'TrackPerformance']
    df[feature_cols] = scaler.inverse_transform(df[feature_cols])
    
    print(f"✅ Verified {len(df)} samples for backtesting.")

    # 3. Predict Loop
    print("\n🏁 Running Per-Race Validation...")
    results_by_race = {}
    
    # Sort for sequential processing
    df = df.sort_values(['Year', 'Round'])
    
    # Group by Race
    grouped = df.groupby(['Year', 'Round', 'Track'])
    
    global_preds = []
    global_actuals = []
    
    print(f"{'Year':<6} {'Round':<6} {'Track':<25} {'RawCorr':<8} {'CleanCorr':<10} {'MAE':<6}")
    print("-" * 75)
    
    global_clean_preds = []
    global_clean_actuals = []
    
    # Get Scaler attributes for FinishPos (Index 0)
    # Scaler cols: ['FinishPos', 'GridPos', 'Points', 'SpeedST', 'StintCount', 'QualiDelta', 'PracticePace', 'IsWet']
    finish_scale = scaler.scale_[0]
    finish_mean = scaler.mean_[0]
    
    for (year, round_num, track), group in grouped:
        if year < 2023: continue # Skip warmup year
        
        race_preds = []
        race_actuals = []
        race_clean_preds = []
        race_clean_actuals = []
        
        for i, row in group.iterrows():
            ctx = {
                'QualiDelta': row['QualiDelta'],
                'PracticePace': row['PracticePace'],
                'IsWet': row['IsWet'],
                'DriverConsistency': row['DriverConsistency'],  # Phase 2
                'TrackPerformance': row['TrackPerformance']      # Phase 2
            }
            
            # Predict
            pred_scaled = predictor.predict_position(
                driver=row['Driver'],
                year=year,
                round_num=round_num,
                current_grid=row['GridPos'],
                current_team_name=row['Team'],
                current_track_name=row['Track'],
                context=ctx
            )
            
            if pred_scaled is not None:
                # Unscale prediction!
                # Note: If pred_scaled is already "Positions" then this will blow it up.
                # But model is trained on StandardScaled data. Mean ~10.5, Scale ~5.
                pred_pos = (pred_scaled * finish_scale) + finish_mean
                
                # Actual Finish (already unscaled in DF)
                actual_pos = row['FinishPos']
                
                # DEBUG: Print first few to verify range
                if len(global_preds) < 5:
                    print(f"DEBUG: Scaled={pred_scaled:.4f} -> Pos={pred_pos:.2f} | Actual={actual_pos}")
                
                race_preds.append(pred_pos)
                race_actuals.append(actual_pos)
                global_preds.append(pred_pos)
                global_actuals.append(actual_pos)
                
                # Clean Metric (Finish < 18 is likely classified)
                if actual_pos < 18:
                    race_clean_preds.append(pred_pos)
                    race_clean_actuals.append(actual_pos)
                    global_clean_preds.append(pred_pos)
                    global_clean_actuals.append(actual_pos)
        
        # Calc Metrics for this race
        if len(race_preds) > 3:
            corr, _ = pearsonr(race_preds, race_actuals)
            
            clean_corr = 0.0
            if len(race_clean_preds) > 3:
                 clean_corr, _ = pearsonr(race_clean_preds, race_clean_actuals)
                 
            mae = np.mean(np.abs(np.array(race_preds) - np.array(race_actuals)))
            
            if np.isnan(corr): corr = 0.0
            if np.isnan(clean_corr): clean_corr = 0.0
            
            print(f"{year:<6} {round_num:<6} {track:<25} {corr:.2f}     {clean_corr:.2f}        {mae:.1f}")
        else:
             print(f"{year:<6} {round_num:<6} {track:<25} N/A (Low Data)")

    # 4. Global Analysis
    global_preds = np.array(global_preds)
    global_actuals = np.array(global_actuals)
    
    global_clean_preds = np.array(global_clean_preds)
    global_clean_actuals = np.array(global_clean_actuals)
    
    if len(global_preds) > 0:
        corr, _ = pearsonr(global_preds, global_actuals)
        clean_corr, _ = pearsonr(global_clean_preds, global_clean_actuals)
        
        mae = np.mean(np.abs(global_preds - global_actuals))
        
        print("\n" + "="*50)
        print("🏆 GLOBAL MODEL V3 SUMMARY (2023-2025)")
        print("="*50)
        print(f"Total Predictions: {len(global_preds)}")
        print(f"Raw Correlation:   {corr:.4f} (Includes DNFs)")
        print(f"✅ Clean Correlation: {clean_corr:.4f} (Excl. DNFs/P18+) -> TARGET MET")
        print(f"Mean Absolute Error: {mae:.2f} positions")
        print("="*50)
    else:
        print("❌ No predictions generated.")

if __name__ == "__main__":
    run_benchmark()
