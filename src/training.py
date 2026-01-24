
import fastf1
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import logging
from src.telemetry import get_telemetry_features

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache setup
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def get_race_data(years=[2023, 2024]):
    """
    Fetch comprehensive race data for training.
    """
    all_data = []
    
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'] == 'conventional'] # Focus on standard races first
            
            logger.info(f"Fetching data for {year} ({len(races)} races)...")
            
            for _, race in races.iterrows():
                if race['EventName'] == 'TOSA': continue # Skip testing
                
                try:
                    round_num = race['RoundNumber']
                    event_name = race['EventName']
                    
                    # Initial check if race happened
                    if year == 2024 and round_num > 24: continue # Adjust as needed
                    
                    logger.info(f"  -> Processing {event_name}")
                    
                    # Load RACE session
                    session_r = fastf1.get_session(year, round_num, 'R')
                    session_r.load(laps=False, telemetry=False, weather=False, messages=False)
                    
                    # Load QUALIFYING session (for telemetry)
                    # We need telemetry here, so we load it. beware of download times.
                    session_q = fastf1.get_session(year, round_num, 'Q')
                    try:
                        session_q.load(laps=True, telemetry=True, weather=False, messages=False)
                        has_qualy = True
                    except Exception as e:
                        logger.warning(f"Failed to load Qualy for {event_name}: {e}")
                        has_qualy = False
                    
                    results = session_r.results
                    if results.empty: continue
                    
                    # Extract features per driver
                    for _, row in results.iterrows():
                        try:
                            grid_pos = row['GridPosition']
                            finish_pos = row['Position']
                            status = row['Status']
                            driver_abbr = row['Abbreviation']
                            
                            # Valid clean race result?
                            is_classified = row['ClassifiedPosition'].isdigit()
                            
                            if grid_pos > 0 and finish_pos > 0 and is_classified:
                                entry = {
                                    'Year': year,
                                    'Round': round_num,
                                    'Track': event_name,
                                    'Driver': driver_abbr,
                                    'Team': row['TeamName'],
                                    'GridPos': grid_pos,
                                    'FinishPos': finish_pos,
                                    'Points': row['Points'],
                                    'Status': status
                                }
                                
                                # Add Telemetry Features from Qualy
                                if has_qualy:
                                    tel_feats = get_telemetry_features(session_q, driver_abbr)
                                    # invalid if empty
                                    if not tel_feats:
                                        # Fill with defaults or skip? 
                                        # Let's fill with NaNs and handle later to avoid dropping too much data
                                        for k in ['Qualy_AvgSpeed', 'Qualy_MaxSpeed', 'Qualy_AvgThrottle', 'Qualy_FullThrottlePct', 'Qualy_AvgBrake', 'Qualy_AvgGear', 'Qualy_GearChanges']:
                                            entry[k] = np.nan
                                    else:
                                        entry.update(tel_feats)
                                else:
                                    for k in ['Qualy_AvgSpeed', 'Qualy_MaxSpeed', 'Qualy_AvgThrottle', 'Qualy_FullThrottlePct', 'Qualy_AvgBrake', 'Qualy_AvgGear', 'Qualy_GearChanges']:
                                        entry[k] = np.nan

                                all_data.append(entry)
                        except:
                            continue
                            
                except Exception as e:
                    logger.warning(f"Failed {event_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
            
    return pd.DataFrame(all_data)

def preprocess_data(df):
    """
    Prepare features for XGBoost.
    """
    print(f"Raw data shape: {df.shape}")
    
    # Feature Engineering
    # 1. Team Power (avg points per race for that team in that year)
    team_perf = df.groupby(['Year', 'Team'])['Points'].mean().reset_index()
    team_perf.rename(columns={'Points': 'TeamAvgPoints'}, inplace=True)
    df = df.merge(team_perf, on=['Year', 'Team'], how='left')
    
    # 2. Track Difficulty (avg DNF rate - simplified proxy)
    # (Skipping complex track features for V1 to ensure robust fit)
    
    # 3. Fill NaNs for Telemetry (using median to be robust)
    tel_cols = ['Qualy_AvgSpeed', 'Qualy_MaxSpeed', 'Qualy_AvgThrottle', 'Qualy_FullThrottlePct', 'Qualy_AvgBrake', 'Qualy_AvgGear', 'Qualy_GearChanges']
    medians = {}
    for col in tel_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            medians[col] = median_val
            
    # Save medians for inference
    with open(os.path.join(MODEL_DIR, 'medians.pkl'), 'wb') as f:
        pickle.dump(medians, f)
    
    # Endcoders
    le_team = LabelEncoder()
    df['TeamId'] = le_team.fit_transform(df['Team'])
    
    le_track = LabelEncoder()
    df['TrackId'] = le_track.fit_transform(df['Track'])
    
    # Save encoders for inference
    with open(os.path.join(MODEL_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump({'team': le_team, 'track': le_track}, f)
        
    return df

def train_model():
    """
    Main training pipeline.
    """
    print("🚀 Starting Training Pipeline...")
    
    # 1. Get Data
    df = get_race_data(years=[2023, 2024])
    if df.empty:
        print("❌ No data found!")
        return
        
    df = preprocess_data(df)
    
    # 2. Setup Features
    features = ['GridPos', 'TeamAvgPoints', 'TrackId', 'TeamId', 'Year',
                'Qualy_AvgSpeed', 'Qualy_MaxSpeed', 'Qualy_AvgThrottle', 
                'Qualy_FullThrottlePct', 'Qualy_AvgBrake', 'Qualy_GearChanges']
    target = 'FinishPos'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. XGBoost Regressor
    # We use a regressor to predict exact position (float) then rank
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='reg:squarederror',
        n_jobs=-1,
        tree_method='hist',
        device='cuda'
    )
    
    print(f"Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\n📊 Model Performance:")
    print(f"   MAE: {mae:.2f} positions")
    print(f"   R2:  {r2:.2f}")
    
    # Feature Importance
    print("\n   Feature Importance:")
    for name, imp in zip(features, model.feature_importances_):
        print(f"   - {name}: {imp:.4f}")
        
    # 5. Save Model
    model.save_model(os.path.join(MODEL_DIR, 'race_predictor.json'))
    print(f"\n✅ Model saved to {MODEL_DIR}/race_predictor.json")

if __name__ == "__main__":
    train_model()
