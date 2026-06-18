import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from src.lstm_predictor import LSTMPredictor
from src.transformer_predictor import TransformerPredictor

def generate_report():
    print("🧠 Loading LSTM Predictor...")
    lstm = LSTMPredictor()
    print("🧠 Loading Transformer Predictor...")
    transformer = TransformerPredictor()
    
    driver_predictions = []
    race_summaries = []
    
    # Evaluate 2024 and 2025 seasons
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
                
            valid_drivers = []
            valid_actuals = []
            lstm_preds = []
            trans_preds = []
            
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
                    
                    trans_pred_scaled = transformer.predict_position(
                        driver=driver,
                        year=year,
                        round_num=round_num,
                        current_grid=grid,
                        current_team_name=row_data['Team'],
                        current_track_name=row_data['Track'],
                        context=ctx
                    )
                    
                    if lstm_pred_scaled is not None and trans_pred_scaled is not None:
                        lstm_pred_val = (lstm_pred_scaled * lstm.scaler.scale_[0]) + lstm.scaler.mean_[0]
                        lstm_pred_val = np.clip(lstm_pred_val, 1.0, 20.0)
                        
                        trans_pred_val = (trans_pred_scaled * transformer.scaler.scale_[0]) + transformer.scaler.mean_[0]
                        trans_pred_val = np.clip(trans_pred_val, 1.0, 20.0)
                        
                        lstm_preds.append(lstm_pred_val)
                        trans_preds.append(trans_pred_val)
                        valid_drivers.append(driver)
                        valid_actuals.append(act)
                        
                        driver_predictions.append({
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
                            'IsWet': row_data.get('IsWet', 0.0),
                            'ReliabilityRisk': row_data.get('ReliabilityRisk', 0.1)
                        })
                except Exception as e:
                    pass
            
            if len(valid_drivers) > 3:
                lstm_rank = pd.Series(lstm_preds).rank().tolist()
                trans_rank = pd.Series(trans_preds).rank().tolist()
                
                corr_lstm, _ = spearmanr(lstm_rank, valid_actuals)
                corr_trans, _ = spearmanr(trans_rank, valid_actuals)
                
                # Mean Absolute Errors
                lstm_mae = np.mean([abs(p - a) for p, a in zip(lstm_preds, valid_actuals)])
                trans_mae = np.mean([abs(p - a) for p, a in zip(trans_preds, valid_actuals)])
                
                race_summaries.append({
                    'Year': year,
                    'Round': round_num,
                    'Race': race_name,
                    'LSTMCorr': corr_lstm if not np.isnan(corr_lstm) else 0.0,
                    'TransCorr': corr_trans if not np.isnan(corr_trans) else 0.0,
                    'LSTM_MAE': lstm_mae,
                    'Trans_MAE': trans_mae,
                    'Drivers': len(valid_drivers)
                })

    df_drivers = pd.DataFrame(driver_predictions)
    df_races = pd.DataFrame(race_summaries)
    
    # Compile Report
    report = []
    report.append("# 🏎️ F1 Deep Learning Predictor: Full Accuracy & Error Report\n")
    report.append("This report breaks down the performance of the **LSTM (V4)** and **Transformer (V5)** models across all analyzed 2024 and 2025 races. Both models are evaluated using **Spearman Rank Correlation** (how well they predict correct standing order) and **Mean Absolute Error (MAE)** in grid positions.\n")
    
    # 1. Overall Metrics
    report.append("## 📊 Overall Performance Summary\n")
    report.append("| Metric | LSTM Predictor | Transformer Predictor |")
    report.append("| :--- | :---: | :---: |")
    report.append(f"| **2024 Avg Spearman Correlation** | {df_races[df_races['Year'] == 2024]['LSTMCorr'].mean():.4f} | {df_races[df_races['Year'] == 2024]['TransCorr'].mean():.4f} |")
    report.append(f"| **2025 Avg Spearman Correlation** | {df_races[df_races['Year'] == 2025]['LSTMCorr'].mean():.4f} | {df_races[df_races['Year'] == 2025]['TransCorr'].mean():.4f} |")
    report.append(f"| **Overall Mean Absolute Error (MAE)** | {df_drivers['LSTMError'].mean():.2f} pos | {df_drivers['TransError'].mean():.2f} pos |")
    report.append(f"| **Dry Weather MAE** | {df_drivers[df_drivers['IsWet'] == 0.0]['LSTMError'].mean():.2f} pos | {df_drivers[df_drivers['IsWet'] == 0.0]['TransError'].mean():.2f} pos |")
    report.append(f"| **Wet Weather MAE** | {df_drivers[df_drivers['IsWet'] == 1.0]['LSTMError'].mean():.2f} pos | {df_drivers[df_drivers['IsWet'] == 1.0]['TransError'].mean():.2f} pos |")
    report.append("\n---\n")
    
    # 2. Top 5 Most Accurate Races (Transformer)
    report.append("## 🏆 Top 5 Most Accurate Races (Transformer)\n")
    report.append("These races achieved the highest rank correlation, indicating near-perfect prediction of the standing order.\n")
    best_races = df_races.sort_values('TransCorr', ascending=False).head(5)
    report.append("| Year | Round | Race | Transformer Correlation | LSTM Correlation | Drivers |")
    report.append("| :---: | :---: | :--- | :---: | :---: | :---: |")
    for _, r in best_races.iterrows():
        report.append(f"| {r['Year']} | {r['Round']} | {r['Race']} | **{r['TransCorr']:.4f}** | {r['LSTMCorr']:.4f} | {r['Drivers']} |")
    report.append("\n---\n")
    
    # 3. Top 5 Least Accurate Races (Transformer)
    report.append("## ⚠️ Top 5 Least Accurate Races (Transformer)\n")
    report.append("These races had the lowest Spearman correlation, representing unexpected race results or model misses.\n")
    worst_races = df_races.sort_values('TransCorr', ascending=True).head(5)
    report.append("| Year | Round | Race | Transformer Correlation | LSTM Correlation | Drivers |")
    report.append("| :---: | :---: | :--- | :---: | :---: | :---: |")
    for _, r in worst_races.iterrows():
        report.append(f"| {r['Year']} | {r['Round']} | {r['Race']} | **{r['TransCorr']:.4f}** | {r['LSTMCorr']:.4f} | {r['Drivers']} |")
    report.append("\n---\n")
    
    # 4. Detail Analysis of Misses
    report.append("## 🔍 Deep Dive: Why Did the Model Miss?\n")
    for _, wr in worst_races.iterrows():
        yr, rnd, r_name = wr['Year'], wr['Round'], wr['Race']
        report.append(f"### 📍 {yr} {r_name} (Trans Correlation: {wr['TransCorr']:.4f})\n")
        report.append("Below are the drivers in this race sorted by largest Transformer prediction errors:\n")
        
        race_drivers = df_drivers[(df_drivers['Year'] == yr) & (df_drivers['Round'] == rnd)]
        worst_drivers = race_drivers.sort_values('TransError', ascending=False).head(5)
        
        report.append("| Driver | Team | Grid | Actual | Predicted | Trans Error | DNF Risk |")
        report.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        for _, d in worst_drivers.iterrows():
            report.append(f"| {d['Driver']} | {d['Team']} | {d['Grid']} | {d['Actual']} | {d['TransPred']:.1f} | **{d['TransError']:.2f}** | {d['ReliabilityRisk']:.2f} |")
        
        # Add analysis text
        reasons = []
        is_wet = "Wet" if race_drivers['IsWet'].iloc[0] > 0 else "Dry"
        reasons.append(f"- **Weather condition**: {is_wet} race.")
        
        # Check if anyone retired (Actual = 20) but was predicted to finish high
        retired_misses = worst_drivers[(worst_drivers['Actual'] >= 18) & (worst_drivers['TransPred'] < 12)]
        if not retired_misses.empty:
            driver_names = ", ".join(retired_misses['Driver'].tolist())
            reasons.append(f"- **Unexpected Retirements**: Drivers like **{driver_names}** retired or fell to the back unexpectedly, creating large gaps between actual and predicted rankings.")
            
        # Check if anyone started low but finished high (Actual < 8, Grid > 12)
        climbers = worst_drivers[(worst_drivers['Actual'] <= 8) & (worst_drivers['Grid'] >= 12)]
        if not climbers.empty:
            driver_names = ", ".join(climbers['Driver'].tolist())
            reasons.append(f"- **Grid Climbers**: **{driver_names}** made massive grid recoveries from the back of the pack, which statistical sequence history did not fully anticipate.")
            
        if not reasons:
            reasons.append("- General midfield pacing variance and pit-stop strategy undercuts.")
            
        report.append("\n**Key factors contributing to the miss:**\n")
        report.append("\n".join(reasons))
        report.append("\n")
        
    report.append("\n---\n")
    report.append("## 📋 Complete Race-by-Race Standing Report\n")
    report.append("| Year | Round | Race | Drivers | LSTM Corr | Trans Corr | LSTM MAE | Trans MAE |")
    report.append("| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_races.sort_values(['Year', 'Round']).iterrows():
        report.append(f"| {r['Year']} | {r['Round']} | {r['Race']} | {r['Drivers']} | {r['LSTMCorr']:.4f} | {r['TransCorr']:.4f} | {r['LSTM_MAE']:.2f} | {r['Trans_MAE']:.2f} |")
        
    # Write to file
    report_content = "\n".join(report)
    
    # Also write to artifact path for UI viewing
    artifact_report_path = r"C:\Users\fadi\.gemini\antigravity-ide\brain\9bf286e0-4159-4d7f-9b2d-705dd46ef3de\race_prediction_report.md"
    with open(artifact_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    report_path = "reports/race_prediction_report.md"
    os.makedirs("reports", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"✅ Full report generated successfully at: {report_path}")

if __name__ == "__main__":
    generate_report()
