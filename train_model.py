"""
train_model.py — Standalone training script for IPL Score Predictor
Run: python train_model.py

NOTE: The trained model still uses pp/mid-over features internally.
      At prediction time (app.py), these are auto-estimated from team/venue
      averages so the user only needs to supply: team, venue, toss.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, 'data')
MODELS = os.path.join(BASE, 'models')
os.makedirs(MODELS, exist_ok=True)

print("Loading data...")
bb     = pd.read_csv(f'{DATA}/ball_by_ball_data.csv')
m      = pd.read_csv(f'{DATA}/ipl_matches_data.csv')
teams  = pd.read_csv(f'{DATA}/teams_data.csv')

m['season_year'] = m['season'].astype(str).str[:4].astype(int)

bb = bb.merge(m[['match_id', 'venue', 'season_year']], on='match_id', how='left')
bb = bb.merge(
    teams[['team_id', 'team_name']].rename(columns={'team_id': 'team_batting', 'team_name': 'batting_team_name'}),
    on='team_batting', how='left')
bb = bb.merge(
    teams[['team_id', 'team_name']].rename(columns={'team_id': 'team_bowling', 'team_name': 'bowling_team_name'}),
    on='team_bowling', how='left')
bb = bb[bb['is_super_over'] == 0]

print(f"Ball-by-ball rows: {len(bb):,}")

# ── Pre-computed stats ──────────────────────────────────────────────────────
innings_scores = bb.groupby(
    ['match_id', 'innings', 'team_batting', 'batting_team_name', 'bowling_team_name', 'venue', 'season_year']
)['total_runs'].sum().reset_index()
innings_scores.rename(columns={'total_runs': 'final_score'}, inplace=True)
innings_scores.to_csv(f'{DATA}/innings_scores.csv', index=False)

batter_stats = bb.groupby('batter').agg(
    career_runs=('batter_runs', 'sum'),
    career_balls=('batter_runs', 'count'),
    career_innings=('match_id', 'nunique')
).reset_index()
batter_stats['career_sr']  = (batter_stats['career_runs'] / batter_stats['career_balls'].clip(1)) * 100
batter_stats['career_avg'] = batter_stats['career_runs'] / batter_stats['career_innings'].clip(1)
batter_stats.to_csv(f'{DATA}/batter_stats.csv', index=False)
print(f"Batter stats: {len(batter_stats)} players")

bowler_stats = bb.groupby('bowler').agg(
    career_wickets=('is_wicket', 'sum'),
    career_runs_conceded=('total_runs', 'sum'),
    career_balls=('total_runs', 'count')
).reset_index()
bowler_stats['career_economy']  = bowler_stats['career_runs_conceded'] / (bowler_stats['career_balls'].clip(1) / 6)
bowler_stats['career_sr_bowl']  = bowler_stats['career_balls'] / bowler_stats['career_wickets'].clip(1)
bowler_stats.to_csv(f'{DATA}/bowler_stats.csv', index=False)
print(f"Bowler stats: {len(bowler_stats)} players")

venue_grp = innings_scores.groupby('venue')['final_score'].agg(['mean', 'std', 'count']).reset_index()
venue_grp.columns = ['venue', 'avg_score', 'std_score', 'matches']
venue_grp.to_csv(f'{DATA}/venue_stats.csv', index=False)

team_stats = innings_scores.groupby('batting_team_name')['final_score'].agg(['mean', 'std', 'count']).reset_index()
team_stats.columns = ['team', 'avg_score', 'std_score', 'innings']
team_stats.to_csv(f'{DATA}/team_bat_stats.csv', index=False)

bvb = bb.groupby(['batter', 'bowler']).agg(
    runs=('batter_runs', 'sum'),
    balls=('batter_runs', 'count'),
    wickets=('is_wicket', 'sum')
).reset_index()
bvb = bvb[bvb['balls'] >= 3]
bvb['sr'] = (bvb['runs'] / bvb['balls'].clip(1)) * 100
bvb.to_csv(f'{DATA}/batter_vs_bowler.csv', index=False)
print(f"Matchup records: {len(bvb):,}")

# ── ML training ─────────────────────────────────────────────────────────────
print("\nBuilding ML training data...")
first_inn = bb[bb['innings'] == 1].copy()
scores    = innings_scores[innings_scores['innings'] == 1]

features = []
for (match_id, team), group in first_inn.groupby(['match_id', 'batting_team_name']):
    group     = group.sort_values(['over_number', 'ball_number'])
    score_row = scores[(scores['match_id'] == match_id) & (scores['batting_team_name'] == team)]
    if len(score_row) == 0:
        continue
    row = score_row.iloc[0]
    pp  = group[group['over_number'] < 6]
    mid = group[group['over_number'] < 11]
    if len(pp) < 15:
        continue
    features.append({
        'batting_team':  team,
        'bowling_team':  row['bowling_team_name'],
        'venue':         row['venue'],
        'season':        int(row['season_year']),
        'pp_runs':       int(pp['total_runs'].sum()),
        'pp_wickets':    int(pp['is_wicket'].sum()),
        'pp_run_rate':   float(pp['total_runs'].sum()) / 6,
        'mid_runs':      int(mid['total_runs'].sum()),
        'mid_wickets':   int(mid['is_wicket'].sum()),
        'final_score':   int(row['final_score'])
    })

df = pd.DataFrame(features)
print(f"Training samples: {len(df)}")

le_team  = LabelEncoder()
le_venue = LabelEncoder()
le_bowl  = LabelEncoder()
df['team_enc']  = le_team.fit_transform(df['batting_team'].fillna('unknown'))
df['venue_enc'] = le_venue.fit_transform(df['venue'].fillna('unknown'))
df['bowl_enc']  = le_bowl.fit_transform(df['bowling_team'].fillna('unknown'))

X = df[['pp_runs', 'pp_wickets', 'pp_run_rate', 'mid_runs', 'mid_wickets',
        'team_enc', 'venue_enc', 'bowl_enc', 'season']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Gradient Boosting Regressor...")
model = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8, random_state=42
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
print(f"  MAE : {mae:.2f} runs")
print(f"  R²  : {r2:.4f}")

# Save
with open(f'{MODELS}/score_model.pkl', 'wb') as f: pickle.dump(model, f)
with open(f'{MODELS}/le_team.pkl',     'wb') as f: pickle.dump(le_team, f)
with open(f'{MODELS}/le_venue.pkl',    'wb') as f: pickle.dump(le_venue, f)
with open(f'{MODELS}/le_bowl.pkl',     'wb') as f: pickle.dump(le_bowl, f)

pd.DataFrame({'team':  sorted(df['batting_team'].dropna().unique())}).to_csv(f'{DATA}/known_teams.csv',  index=False)
pd.DataFrame({'venue': sorted(df['venue'].dropna().unique())       }).to_csv(f'{DATA}/known_venues.csv', index=False)

print("\n✅ Model and all stats saved successfully!")
print(f"   Models : {MODELS}/")
print(f"   Stats  : {DATA}/")
