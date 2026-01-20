import fastf1
import pandas as pd
from scipy.stats import spearmanr

def evaluate_sim_performance(year, race_name, results_agg):
    """
    Automated Auditor: Compares results_agg directly with FIA data.
    """
    try:
        # 1. Fetch Actual Results
        session = fastf1.get_session(year, race_name, 'R')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        actual = session.results[['Abbreviation', 'ClassifiedPosition', 'GridPosition']]
        
        # 2. Build Comparison Table
        comparison = []
        for i, pred in enumerate(results_agg):
            driver = pred['Driver']
            actual_row = actual[actual['Abbreviation'] == driver]
            
            if not actual_row.empty:
                # Convert 'R' (Retired) or 'D' (Disqualified) to P20 for math
                try:
                    act_pos = int(actual_row['ClassifiedPosition'].values[0])
                except:
                    act_pos = 20
                
                comparison.append({
                    'Driver': driver,
                    'Predicted': i + 1, # Rank in results_agg
                    'Actual': act_pos,
                    'Start': int(actual_row['GridPosition'].values[0])
                })
        
        df = pd.DataFrame(comparison)
        
        # 3. Calculate Correlation
        # We use Spearman to see if the 'order' of finishers matches.
        # 
        corr, _ = spearmanr(df['Predicted'], df['Actual'])
        
        # 4. Podium Accuracy
        top3_pred = set(df.nsmallest(3, 'Predicted')['Driver'])
        top3_act = set(df.nsmallest(3, 'Actual')['Driver'])
        podium_hits = len(top3_pred.intersection(top3_act))
        
        return {
            'correlation': corr,
            'podium_accuracy': podium_hits,
            'data': df
        }
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return None