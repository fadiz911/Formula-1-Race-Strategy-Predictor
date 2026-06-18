import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.lstm_predictor import LSTMPredictor
from src.transformer_predictor import TransformerPredictor

def run_evaluation():
    print("🧠 Loading LSTM Predictor...")
    lstm = LSTMPredictor()
    print("🧠 Loading Transformer Predictor...")
    transformer = TransformerPredictor()
    
    if not lstm.loaded:
        print("❌ LSTM not loaded!")
    if not transformer.loaded:
        print("❌ Transformer not loaded!")
        
    results = []
    
    # We will evaluate 2024 and 2025 seasons
    for year in [2024, 2025]:
        summary_path = f"reports/season_{year}/{year}_season_summary.csv"
        if not os.path.exists(summary_path):
            print(f"⚠️ Summary for {year} does not exist. Skipping.")
            continue
            
        df_summary = pd.read_csv(summary_path)
        eval_dir = f"reports/season_{year}/evaluations/"
        
        for idx, row_summary in df_summary.iterrows():
            round_num = int(row_summary['Round'])
            race_name = row_summary['Race']
            
            # Find the evaluation file for this race
            # format: {year}_R{round:02d}_{race_name_under}_evaluation.csv
            race_under = race_name.replace(" ", "_")
            eval_file = f"{year}_R{round_num:02d}_{race_under}_evaluation.csv"
            eval_path = os.path.join(eval_dir, eval_file)
            
            if not os.path.exists(eval_path):
                continue
                
            df_eval = pd.read_csv(eval_path)
            if df_eval.empty:
                continue
                
            # We want to run LSTM and Transformer predictions for all drivers in df_eval
            # df_eval contains: Year, Round, Race, Driver, Start, Actual, Predicted, Misplacement, AbsMisplacement
            
            # Load raw history context for features if needed
            # For LSTM: we can use LSTMPredictor's internal history_df
            # Let's extract context from history_df or build it
            drivers = df_eval['Driver'].tolist()
            actuals = df_eval['Actual'].tolist()
            physics_predicted = df_eval['Predicted'].tolist()
            
            lstm_preds = []
            trans_preds = []
            valid_drivers = []
            valid_actuals = []
            valid_physics = []
            
            for driver, act, phys in zip(drivers, actuals, physics_predicted):
                # We need to construct context.
                # In LSTMPredictor history_df, we have the features for this driver/race
                # Let's find the row in history_df
                # Note: history_df is scaled, raw_history_df is unscaled
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
                        current_grid=row_data['GridPos'],
                        current_team_name=row_data['Team'],
                        current_track_name=row_data['Track'],
                        context=ctx
                    )
                    
                    trans_pred = transformer.predict_position(
                        driver=driver,
                        year=year,
                        round_num=round_num,
                        current_grid=row_data['GridPos'],
                        current_team_name=row_data['Team'],
                        current_track_name=row_data['Track'],
                        context=ctx
                    )
                    
                    if lstm_pred_scaled is not None and trans_pred is not None:
                        # Unscale predictions
                        lstm_pred_val = (lstm_pred_scaled * lstm.scaler.scale_[0]) + lstm.scaler.mean_[0]
                        trans_pred_val = (trans_pred * transformer.scaler.scale_[0]) + transformer.scaler.mean_[0]
                        
                        lstm_preds.append(lstm_pred_val)
                        trans_preds.append(trans_pred_val)
                        valid_drivers.append(driver)
                        valid_actuals.append(act)
                        valid_physics.append(phys)
                except Exception as e:
                    pass
            
            if len(valid_drivers) > 3:
                # Rank predicted positions
                physics_rank = pd.Series(valid_physics).rank().tolist()
                lstm_rank = pd.Series(lstm_preds).rank().tolist()
                trans_rank = pd.Series(trans_preds).rank().tolist()
                
                # Try blends
                blend_73 = (np.array(lstm_preds) * 0.7 + np.array(trans_preds) * 0.3).tolist()
                blend_82 = (np.array(lstm_preds) * 0.8 + np.array(trans_preds) * 0.2).tolist()
                blend_55 = (np.array(lstm_preds) * 0.5 + np.array(trans_preds) * 0.5).tolist()
                
                blend_73_rank = pd.Series(blend_73).rank().tolist()
                blend_82_rank = pd.Series(blend_82).rank().tolist()
                blend_55_rank = pd.Series(blend_55).rank().tolist()
                
                corr_phys, _ = spearmanr(physics_rank, valid_actuals)
                corr_lstm, _ = spearmanr(lstm_rank, valid_actuals)
                corr_trans, _ = spearmanr(trans_rank, valid_actuals)
                corr_73, _ = spearmanr(blend_73_rank, valid_actuals)
                corr_82, _ = spearmanr(blend_82_rank, valid_actuals)
                corr_55, _ = spearmanr(blend_55_rank, valid_actuals)
                
                results.append({
                    'Year': year,
                    'Round': round_num,
                    'Race': race_name,
                    'PhysCorr': corr_phys if not np.isnan(corr_phys) else 0.0,
                    'LSTMCorr': corr_lstm if not np.isnan(corr_lstm) else 0.0,
                    'TransCorr': corr_trans if not np.isnan(corr_trans) else 0.0,
                    'Blend73Corr': corr_73 if not np.isnan(corr_73) else 0.0,
                    'Blend82Corr': corr_82 if not np.isnan(corr_82) else 0.0,
                    'Blend55Corr': corr_55 if not np.isnan(corr_55) else 0.0,
                    'Drivers': len(valid_drivers)
                })
                
    df_res = pd.DataFrame(results)
    
    # Save the comparison summary
    df_res.to_csv("reports/predictor_comparison_summary.csv", index=False)
    
    print("\n=======================================================")
    print("🏆 COMPARATIVE ANALYSIS SUMMARY (2024 & 2025)")
    print("=======================================================")
    print(df_res.to_markdown(index=False))
    
    print("\nAVERAGE CORRELATIONS BY SEASON:")
    for yr in [2024, 2025]:
        yr_data = df_res[df_res['Year'] == yr]
        print(f"\n--- {yr} Season (Rounds: {len(yr_data)}) ---")
        print(f"Physics Simulation: {yr_data['PhysCorr'].mean():.4f}")
        print(f"LSTM Predictor (DL):  {yr_data['LSTMCorr'].mean():.4f}")
        print(f"Transformer (DL):    {yr_data['TransCorr'].mean():.4f}")
        print(f"70/30 Ensemble Blend: {yr_data['Blend73Corr'].mean():.4f}")
        print(f"80/20 Ensemble Blend: {yr_data['Blend82Corr'].mean():.4f}")
        print(f"50/55 Ensemble Blend: {yr_data['Blend55Corr'].mean():.4f}")
    print("=======================================================")

if __name__ == "__main__":
    run_evaluation()
