from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))

# Load model and encoders
with open(f'{BASE}/models/score_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'{BASE}/models/le_team.pkl', 'rb') as f:
    le_team = pickle.load(f)
with open(f'{BASE}/models/le_venue.pkl', 'rb') as f:
    le_venue = pickle.load(f)
with open(f'{BASE}/models/le_bowl.pkl', 'rb') as f:
    le_bowl = pickle.load(f)

# Load stats
team_stats    = pd.read_csv(f'{BASE}/data/team_bat_stats.csv')
venue_stats   = pd.read_csv(f'{BASE}/data/venue_stats.csv')
innings_scores = pd.read_csv(f'{BASE}/data/innings_scores.csv')
batter_stats  = pd.read_csv(f'{BASE}/data/batter_stats.csv')
bowler_stats  = pd.read_csv(f'{BASE}/data/bowler_stats.csv')
bvb           = pd.read_csv(f'{BASE}/data/batter_vs_bowler.csv')
known_teams   = pd.read_csv(f'{BASE}/data/known_teams.csv')['team'].tolist()
known_venues  = pd.read_csv(f'{BASE}/data/known_venues.csv')['venue'].tolist()

# Pre-compute head-to-head win counts from innings_scores + match results
matches = pd.read_csv(f'{BASE}/data/ipl_matches_data.csv')
teams_ref = pd.read_csv(f'{BASE}/data/teams_data.csv').set_index('team_id')['team_name'].to_dict()
matches['t1_name'] = matches['team1'].map(teams_ref)
matches['t2_name'] = matches['team2'].map(teams_ref)
matches['winner_name'] = matches['match_winner'].map(teams_ref)

h2h_records = {}
for _, row in matches.dropna(subset=['t1_name', 't2_name', 'winner_name']).iterrows():
    key = tuple(sorted([row['t1_name'], row['t2_name']]))
    h2h_records.setdefault(key, {})
    w = row['winner_name']
    h2h_records[key][w] = h2h_records[key].get(w, 0) + 1


def safe_encode(le, val):
    try:
        return le.transform([val])[0]
    except Exception:
        return 0


def get_team_avg(team):
    row = team_stats[team_stats['team'] == team]
    return float(row['avg_score'].values[0]) if len(row) else 160.0


def get_venue_avg(venue):
    row = venue_stats[venue_stats['venue'] == venue]
    return float(row['avg_score'].values[0]) if len(venue_rows := venue_stats[venue_stats['venue'] == venue]) else 160.0


def estimate_pp_mid(bat_avg, venue_avg, bowl_avg):
    """
    Estimate plausible powerplay and mid-over values from historical averages.
    PP typically contributes ~35% of total, mid (overs 1-10) ~55%.
    """
    base = (bat_avg * 0.5 + venue_avg * 0.35 + (165 - (bowl_avg - 160) * 0.3) * 0.15)
    pp_runs    = round(base * 0.34)
    pp_wickets = 1
    mid_runs   = round(base * 0.54)
    mid_wickets = 3
    return pp_runs, pp_wickets, mid_runs, mid_wickets


@app.route('/')
def index():
    return render_template('index.html',
                           teams=known_teams,
                           venues=known_venues)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Send a JSON object in the request body.'}), 400
    batting_team = data.get('batting_team', '')
    bowling_team = data.get('bowling_team', '')
    venue        = data.get('venue', '')
    toss_winner  = data.get('toss_winner', '')
    toss_decision = data.get('toss_decision', 'bat')
    try:
        season = int(data.get('season', 2024))
    except (TypeError, ValueError):
        return jsonify({'error': 'season must be a whole number, for example 2024.'}), 400

    # Get historical averages
    bat_avg  = get_team_avg(batting_team)
    bowl_avg = get_team_avg(bowling_team)
    v_row    = venue_stats[venue_stats['venue'] == venue]
    ven_avg  = float(v_row['avg_score'].values[0]) if len(v_row) else 160.0

    # Auto-estimate pp/mid features from averages
    pp_runs, pp_wickets, mid_runs, mid_wickets = estimate_pp_mid(bat_avg, ven_avg, bowl_avg)

    # Encode
    team_enc  = safe_encode(le_team, batting_team)
    venue_enc = safe_encode(le_venue, venue)
    bowl_enc  = safe_encode(le_bowl, bowling_team)

    pp_run_rate = pp_runs / 6.0
    features = np.array([[pp_runs, pp_wickets, pp_run_rate,
                          mid_runs, mid_wickets,
                          team_enc, venue_enc, bowl_enc, season]])
    predicted = float(model.predict(features)[0])

    # Toss adjustment (small nudge)
    if toss_winner == batting_team:
        predicted += 3 if toss_decision == 'bat' else -1
    else:
        predicted += -3 if toss_decision == 'field' else 1

    predicted = round(predicted)
    low  = round(predicted * 0.94)
    high = round(predicted * 1.06)

    # Head-to-head
    key = tuple(sorted([batting_team, bowling_team]))
    h2h = h2h_records.get(key, {})
    bat_wins  = h2h.get(batting_team, 0)
    bowl_wins = h2h.get(bowling_team, 0)
    total_h2h = bat_wins + bowl_wins

    # Win probability
    h2h_prob   = (bat_wins / total_h2h) if total_h2h > 0 else 0.5
    bowl_std   = float(team_stats[team_stats['team'] == bowling_team]['std_score'].values[0]) if bowling_team in team_stats['team'].values else 30
    score_adv  = (predicted - bowl_avg) / max(bowl_std, 1)
    score_prob = min(max(0.5 + score_adv * 0.15, 0.25), 0.75)
    toss_prob  = (0.53 if toss_decision == 'bat' else 0.48) if toss_winner == batting_team else (0.47 if toss_decision == 'field' else 0.52)

    bat_prob = h2h_prob * 0.35 + score_prob * 0.45 + toss_prob * 0.20
    bat_prob = min(max(bat_prob, 0.20), 0.80)
    bat_pct  = round(bat_prob * 100)
    bowl_pct = 100 - bat_pct

    winner     = batting_team if bat_pct >= 50 else bowling_team
    winner_pct = bat_pct if bat_pct >= 50 else bowl_pct

    return jsonify({
        'predicted_score': predicted,
        'range_low':  low,
        'range_high': high,
        'bat_win_pct':  bat_pct,
        'bowl_win_pct': bowl_pct,
        'winner':       winner,
        'winner_pct':   winner_pct,
        'team_avg':     round(bat_avg),
        'venue_avg':    round(ven_avg),
        'h2h': {'bat_wins': bat_wins, 'bowl_wins': bowl_wins, 'total': total_h2h},
    })


# ── Existing stat endpoints (unchanged) ─────────────────────────────────────

@app.route('/api/batter_stats')
def get_batter_stats():
    name = request.args.get('name', '').strip()
    row = batter_stats[batter_stats['batter'].str.lower() == name.lower()]
    if len(row) == 0:
        row = batter_stats[batter_stats['batter'].str.lower().str.contains(name.lower(), na=False)].head(5)
    if len(row) == 0:
        return jsonify({'error': 'Player not found'})
    return jsonify(row[['batter', 'career_runs', 'career_balls', 'career_innings', 'career_sr', 'career_avg']].head(5).to_dict(orient='records'))


@app.route('/api/bowler_stats')
def get_bowler_stats():
    name = request.args.get('name', '').strip()
    row = bowler_stats[bowler_stats['bowler'].str.lower() == name.lower()]
    if len(row) == 0:
        row = bowler_stats[bowler_stats['bowler'].str.lower().str.contains(name.lower(), na=False)].head(5)
    if len(row) == 0:
        return jsonify({'error': 'Player not found'})
    return jsonify(row[['bowler', 'career_wickets', 'career_economy', 'career_sr_bowl', 'career_runs_conceded']].head(5).to_dict(orient='records'))


@app.route('/api/bowler_vs_batter')
def get_bvb():
    bowler = request.args.get('bowler', '').strip()
    row = bvb[bvb['bowler'].str.lower().str.contains(bowler.lower(), na=False)]
    row = row.sort_values('balls', ascending=False).head(10)
    return jsonify(row[['batter', 'bowler', 'runs', 'balls', 'wickets', 'sr']].to_dict(orient='records'))


@app.route('/api/batter_vs_bowler')
def get_bvb2():
    batter = request.args.get('batter', '').strip()
    row = bvb[bvb['batter'].str.lower().str.contains(batter.lower(), na=False)]
    row = row.sort_values('balls', ascending=False).head(10)
    return jsonify(row[['batter', 'bowler', 'runs', 'balls', 'wickets', 'sr']].to_dict(orient='records'))


@app.route('/api/venue_stats')
def get_venue_stats():
    venue = request.args.get('venue', '').strip()
    row = venue_stats[venue_stats['venue'].str.lower().str.contains(venue.lower(), na=False)].head(5)
    return jsonify(row.round(1).to_dict(orient='records'))


@app.route('/api/team_history')
def get_team_history():
    team = request.args.get('team', '').strip()
    inn  = innings_scores[innings_scores['batting_team_name'] == team].copy()
    inn  = inn.sort_values('season_year', ascending=False)
    agg  = inn.groupby('season_year')['final_score'].agg(['mean', 'max', 'min', 'count']).reset_index()
    agg.columns = ['season', 'avg', 'max', 'min', 'matches']
    return jsonify(agg.round(1).to_dict(orient='records'))


@app.route('/api/search_players')
def search_players():
    q       = request.args.get('q', '').strip().lower()
    batters = batter_stats[batter_stats['batter'].str.lower().str.contains(q, na=False)]['batter'].head(8).tolist()
    bowlers = bowler_stats[bowler_stats['bowler'].str.lower().str.contains(q, na=False)]['bowler'].head(8).tolist()
    return jsonify(list(set(batters + bowlers))[:10])


if __name__ == '__main__':
    # Debug mode exposes an interactive debugger, so it is off unless you ask for it:
    #   FLASK_DEBUG=1 python app.py
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', port=5000)
