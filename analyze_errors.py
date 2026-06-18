import os
import pandas as pd
import numpy as np
from src.lstm_predictor import LSTMPredictor
from src.transformer_predictor import TransformerPredictor

def run_error_analysis():
    print("🧠 Loading LSTM Predictor...")
    lstm = LSTMPredictor()
    print("🧠 Loading Transformer Predictor...")
    transformer = TransformerPredictor()
    
    errors = []
    
    # We will evaluate 2024 and 2025 seasons
    for year in [2024, 2025]:
        summary_path = f"reports/season_{year}/{year}_season_summary.csv"
        if not os.path.exists(summary_path):
            continue
            
        df_summary = pd.read_csv(summary_path)
        eval_dir = f"reports/season_{year}/evaluations/"
        
        for idx, row_summary in df_summary.iterrows():
            round_num = int(row_summary['Round'])
            race_name = row_summary['Race']
            
            race_under = race_name.replace(" ", "_")
            eval_file = f"{year}_R{round_num:02d}_{race_under}_evaluation.csv"
            eval_path = os.path.join(eval_dir, eval_file)
            
            if not os.path.exists(eval_path):
                continue
                
            df_eval = pd.read_csv(eval_path)
            if df_eval.empty:
                continue
                
            for _, row_eval in df_eval.iterrows():
                driver = row_eval['Driver']
                act = row_eval['Actual']
                grid = row_eval['Start']
                
                try:
                    raw_hist = lstm.raw_history_df
                    row_data = raw_hist[
                        (raw_hist['Driver'] == driver) &
                        (raw_hist['Year'].round().astype(int) == year) &
                        (raw_hist['Round'].round().astype(int) == round_num)
                    ]
                    
                    if row_data.empty:
                        continue
                        
                    row_data = row_data.iloc[0]
                    
                    # Shared Context
                    ctx = {
                        'QualiDelta': row_data.get('QualiDelta', 2.0),
                        'PracticePace': row_data.get('PracticePace', 0.5),
                        'IsWet': row_data.get('IsWet', 0.0),
                        'DriverConsistency': row_data.get('DriverConsistency', 5.0),
                        'TrackPerformance': row_data.get('TrackPerformance', 10.5),
                        'TeamAvgPoints': row_data.get('TeamAvgPoints', 5.0),
                        'ReliabilityRisk': row_data.get('ReliabilityRisk', 0.1)
                    }
                    
                    lstm_pred_scaled = lstm.predict_position(
                        driver=driver,
                        year=year,
                        round_num=round_num,
                        current_grid=grid,
                        current_team_name=row_data['Team'],
                        current_track_name=row_data['Track'],
                        context=ctx
                    )
                    
                    trans_pred = transformer.predict_position(
                        driver=driver,
                        year=year,
                        round_num=round_num,
                        current_grid=grid,
                        current_team_name=row_data['Team'],
                        current_track_name=row_data['Track'],
                        context=ctx
                    )
                    
                    if lstm_pred_scaled is not None and trans_pred is not None:
                        lstm_pred_val = (lstm_pred_scaled * lstm.scaler.scale_[0]) + lstm.scaler.mean_[0]
                        lstm_pred_val = np.clip(lstm_pred_val, 1.0, 20.0)
                        
                        trans_pred_val = (trans_pred * transformer.scaler.scale_[0]) + transformer.scaler.mean_[0]
                        trans_pred_val = np.clip(trans_pred_val, 1.0, 20.0)
                        
                        errors.append({
                            'Year': year,
                            'Round': round_num,
                            'Race': race_name,
                            'Driver': driver,
                            'Team': row_data['Team'],
                            'Grid': grid,
                            'Actual': act,
                            'LSTMPred': lstm_pred_val,
                            'TransPred': trans_pred_val,
                            'LSTMError': abs(lstm_pred_val - act),
                            'TransError': abs(trans_pred_val - act),
                            'IsWet': row_data.get('IsWet', 0.0)
                        })
                except Exception as e:
                    pass
                    
    df_err = pd.DataFrame(errors)
    
    # Analyze LSTM Worst Errors
    print("\n=======================================================")
    print("🚨 TOP 10 LARGEST LSTM PREDICTION ERRORS:")
    print("=======================================================")
    lstm_worst = df_err.sort_values('LSTMError', ascending=False).head(10)
    print(lstm_worst[['Year', 'Race', 'Driver', 'Grid', 'Actual', 'LSTMPred', 'LSTMError', 'IsWet']].to_markdown(index=False))
    
    # Analyze Transformer Worst Errors
    print("\n=======================================================")
    print("🚨 TOP 10 LARGEST TRANSFORMER PREDICTION ERRORS:")
    print("=======================================================")
    trans_worst = df_err.sort_values('TransError', ascending=False).head(10)
    print(trans_worst[['Year', 'Race', 'Driver', 'Grid', 'Actual', 'TransPred', 'TransError', 'IsWet']].to_markdown(index=False))
    
    # Error under Wet vs Dry conditions
    print("\n=======================================================")
    print("🌧️ WET VS DRY WEATHER PERFORMANCE (MAE):")
    print("=======================================================")
    for condition, name in [(0.0, 'Dry'), (1.0, 'Wet')]:
        sub_df = df_err[df_err['IsWet'] == condition]
        if not sub_df.empty:
            print(f"{name} races ({len(sub_df)} samples):")
            print(f"  - LSTM Mean Error:        {sub_df['LSTMError'].mean():.2f} positions")
            print(f"  - Transformer Mean Error: {sub_df['TransError'].mean():.2f} positions")
            
    # Error by Driver (Who is hardest to predict?)
    print("\n=======================================================")
    print("🏎️ HARDEST DRIVERS TO PREDICT (Top 5 by LSTM MAE):")
    print("=======================================================")
    driver_mae = df_err.groupby('Driver')[['LSTMError', 'TransError']].agg(['mean', 'count'])
    # filter for drivers with at least 3 races
    driver_mae = driver_mae[driver_mae[('LSTMError', 'count')] >= 3]
    driver_mae_sorted = driver_mae.sort_values(('LSTMError', 'mean'), ascending=False).head(5)
    print(driver_mae_sorted.to_markdown())

if __name__ == "__main__":
    run_error_analysis()
