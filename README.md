# 🏏 IPL Score Predictor — ML Project

A machine learning web app that predicts IPL first innings scores and match winner
using **Gradient Boosting Regression** trained on 278,000+ ball-by-ball records (2008–2025).

---

## ✨ What Changed (v2)

- **Simplified inputs** — you only need to select **batting team, bowling team, venue, and toss**.
  No more powerplay sliders or mid-over inputs required.
- **Winner prediction** — every prediction now includes a win probability bar
  based on head-to-head record, score advantage and toss factor.
- **Auto-estimation** — the app internally estimates powerplay/mid-over values
  from historical team & venue averages before feeding the ML model, so you get
  the same GBR accuracy without manual inputs.
- **Cleaner UI** — dark-themed, fully responsive interface with colour-coded badges.

---

## 📊 Datasets Used

| File | Description |
|------|-------------|
| `ball_by_ball_data.csv` | 278K+ delivery records — primary training data |
| `ipl_matches_data.csv`  | 1,169 match records with venue, season, result |
| `players-data-updated.csv` | 772 player profiles |
| `teams_data.csv`        | Team IDs and names |
| `team_aliases.csv`      | Team name aliases across seasons |

---

## 🤖 ML Model

- **Algorithm:** Gradient Boosting Regressor (sklearn)
- **Training samples:** ~1,169 first-innings records
- **Features (internal):** pp_runs, pp_wickets, pp_run_rate, mid_runs, mid_wickets, team, venue, bowling_team, season
- **User inputs required:** batting team, bowling team, venue, toss winner, toss decision
- **Performance:** MAE ≈ 16 runs, R² ≈ 0.56

---

## 🚀 How to Run

```bash
# 1. Unzip the project
unzip ipl_score_predictor_v2.zip
cd ipl_score_predictor

# 2. (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (only if models/ folder is missing)
python train_model.py

# 5. Run the web app
python app.py

# 6. Open in browser
# http://localhost:5000
```

---

## 🖥️ Features

### Score & Winner Prediction
- Select batting team, bowling team, venue, season
- Pick toss winner and decision (bat / bowl)
- Get predicted 1st innings score with confidence range
- Win probability bar with head-to-head breakdown

### Player Stats
- Search any batter: career runs, average, strike rate
- Search any bowler: wickets, economy, bowling SR

### Bowler vs Batter Matchup
- Colour-coded strike rates (green = high, red = low)

### Venue Analysis
- Average scores, std deviation, match count per ground

### Team History
- Season-by-season batting averages with max/min

---

## 📁 Project Structure

```
ipl_score_predictor/
├── app.py                    # Flask server + API (simplified inputs)
├── train_model.py            # Model training script
├── requirements.txt
├── README.md
├── data/
│   ├── ball_by_ball_data.csv
│   ├── ipl_matches_data.csv
│   ├── players-data-updated.csv
│   ├── teams_data.csv
│   ├── team_aliases.csv
│   ├── batter_stats.csv
│   ├── bowler_stats.csv
│   ├── venue_stats.csv
│   ├── team_bat_stats.csv
│   ├── batter_vs_bowler.csv
│   └── innings_scores.csv
├── models/
│   ├── score_model.pkl
│   ├── le_team.pkl
│   ├── le_bowl.pkl
│   └── le_venue.pkl
└── templates/
    └── index.html            # Dark-themed responsive UI
```

---

## 🔧 Retrain the Model

```bash
python train_model.py
```

This regenerates all stats CSVs and re-trains the model from scratch.
