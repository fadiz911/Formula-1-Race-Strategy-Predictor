import fastf1
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache setup
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

MODEL_DIR = 'models_dl'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

class RaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RacePredictorNet(nn.Module):
    def __init__(self, num_teams, num_tracks, embedding_dim=8):
        super(RacePredictorNet, self).__init__()
        
        # Embeddings
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)
        
        # Additional features: GridPos, RollingTeamPoints, DriverForm, Year (normalized)
        input_dim = (embedding_dim * 2) + 4 
        
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, team_idx, track_idx, numerical_features):
        team_emb = self.team_embedding(team_idx)
        track_emb = self.track_embedding(track_idx)
        
        # Concatenate: [TeamEmb, TrackEmb, NumFeatures]
        x = torch.cat([team_emb, track_emb, numerical_features], dim=1)
        
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.output(x)
        return x

def get_race_data(years=[2023, 2024]):
    """
    Fetch comprehensive race data.
    """
    all_data = []
    
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'] == 'conventional']
            
            logger.info(f"Fetching data for {year}...")
            
            for _, race in races.iterrows():
                if race['EventName'] == 'TOSA': continue
                
                try:
                    round_num = race['RoundNumber']
                    event_name = race['EventName']
                    if year == 2024 and round_num > 24: continue

                    session = fastf1.get_session(year, round_num, 'R')
                    session.load(laps=False, telemetry=False, weather=False, messages=False)
                    
                    results = session.results
                    if results.empty: continue
                    
                    for _, row in results.iterrows():
                        try:
                            if row['GridPosition'] > 0 and row['Position'] > 0 and str(row['ClassifiedPosition']).isdigit():
                                all_data.append({
                                    'Year': year,
                                    'Round': round_num,
                                    'Track': event_name,
                                    'Driver': row['Abbreviation'],
                                    'Team': row['TeamName'],
                                    'GridPos': row['GridPosition'],
                                    'FinishPos': row['Position'],
                                    'Points': row['Points']
                                })
                        except: continue
                except Exception as e:
                    print(f"Skipping {race['EventName']}: {e}")
                    
        except Exception as e:
            logger.error(f"Error year {year}: {e}")
            
    return pd.DataFrame(all_data)

def calculate_rolling_features(df):
    """
    Computes rolling average points for teams to capture in-season trends.
    """
    print("Feature Engineering: Calculating Rolling Team Performance...")
    
    df = df.sort_values(['Year', 'Round'])
    
    # Calculate Team Points per Race
    team_race_points = df.groupby(['Year', 'Round', 'Team'])['Points'].sum().reset_index()
    
    # Rolling Window (Last 5 races)
    # We group by Team and shift to ensure we only use PAST data
    team_race_points['RollingPoints'] = team_race_points.groupby('Team')['Points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Fill NaN (first race of season/team) with 0 or slight positive/negative bias
    team_race_points['RollingPoints'] = team_race_points['RollingPoints'].fillna(0)
    
    # Merge back
    df = df.merge(team_race_points[['Year', 'Round', 'Team', 'RollingPoints']], on=['Year', 'Round', 'Team'], how='left')
    return df

def train_dl_model():
    print("🚀 Starting Deep Learning Training Pipeline...")
    
    # 1. Data
    df = get_race_data()
    if df.empty: return
    
    df = calculate_rolling_features(df)
    
    # 2. Encoders
    le_team = LabelEncoder()
    df['TeamId'] = le_team.fit_transform(df['Team'])
    
    le_track = LabelEncoder()
    df['TrackId'] = le_track.fit_transform(df['Track'])
    
    # 3. Scalers
    scaler = StandardScaler()
    # Features: GridPos, RollingPoints, Year (normalized), Round (normalized)
    # Using Year/Round helps capture 'late season' vs 'early season' dynamics implicitly
    num_cols = ['GridPos', 'RollingPoints', 'Year', 'Round']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Save artifacts
    with open(os.path.join(MODEL_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump({
            'team': le_team, 
            'track': le_track, 
            'scaler': scaler,
            'num_teams': len(le_team.classes_),
            'num_tracks': len(le_track.classes_)
        }, f)

    # 4. Tensors
    X_cat = df[['TeamId', 'TrackId']].values
    X_num = df[num_cols].values
    y = df['FinishPos'].values
    
    # Combine for dataset split (hacky but works)
    # Actually, let's keep them separate in the dataset or creating a custom one
    # For simplicity, we pass indices.
    
    # Better: Split DF first
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
    def df_to_tensor(d):
        t_team = torch.tensor(d['TeamId'].values, dtype=torch.long)
        t_track = torch.tensor(d['TrackId'].values, dtype=torch.long)
        t_num = torch.tensor(d[num_cols].values, dtype=torch.float32)
        t_y = torch.tensor(d['FinishPos'].values, dtype=torch.float32).view(-1, 1)
        return t_team, t_track, t_num, t_y
        
    t_team_train, t_track_train, t_num_train, y_train = df_to_tensor(df_train)
    t_team_test, t_track_test, t_num_test, y_test = df_to_tensor(df_test)
    
    # 5. Model
    model = RacePredictorNet(len(le_team.classes_), len(le_track.classes_))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 6. Train Loop
    epochs = 300 # deep learning needs more epochs than boosting ;)
    print(f"Training for {epochs} epochs...")
    
    batch_size = 32
    num_batches = len(y_train) // batch_size
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle (simplified)
        perm = torch.randperm(len(y_train))
        
        for i in range(0, len(y_train), batch_size):
            indices = perm[i:i+batch_size]
            b_team = t_team_train[indices]
            b_track = t_track_train[indices]
            b_num = t_num_train[indices]
            b_y = y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(b_team, b_track, b_num)
            loss = criterion(outputs, b_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(t_team_test, t_track_test, t_num_test)
                val_loss = criterion(val_out, y_test)
                print(f"Epoch {epoch}: Train Loss {epoch_loss/num_batches:.4f} | Val MSE {val_loss:.4f}")

    # Save
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'model.pth'))
    print("✅ PyTrack Deep Learning Model Saved!")

if __name__ == "__main__":
    train_dl_model()
