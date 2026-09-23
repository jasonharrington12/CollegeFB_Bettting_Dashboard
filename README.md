# 🏈 College Football Betting Dashboard

A Streamlit app that rates FBS teams from game results and compares its predictions
to sportsbook lines for **spreads, moneylines, and over/unders**. For each bet it shows
the model win probability, expected value (EV), and a fractional-Kelly stake.

> For entertainment and learning. A simple ratings model rarely beats the closing line.
> Only risk money you can afford to lose.

## How it works
1. Pulls finished games from the [CollegeFootballData](https://collegefootballdata.com) API.
2. Fits a ridge-regression points model: each team gets an offense and a defense rating,
   plus a home-field advantage. Non-FBS opponents are grouped into one bucket.
3. Predicts a score, spread, total, and win probability for each game.
4. Compares those to market lines and computes EV for each bet.

## Tabs
- **Best Bets** – bets above your minimum EV, sorted by EV
- **Full Slate** – model vs. market for every game that week
- **Matchup Explorer** – pick any two teams and enter your own book's lines
- **Team Ratings** – offense, defense, and overall ratings

## Run it locally
```bash
git clone https://github.com/YOUR-USERNAME/cfb-betting-dashboard.git
cd cfb-betting-dashboard
pip install -r requirements.txt
```

Get a free API key at <https://collegefootballdata.com/key>, then either paste it into the
app's sidebar, or save it locally:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit secrets.toml and add your key
```

Start the app:
```bash
streamlit run cfb_dashboard.py
```

## Deploy on Streamlit Community Cloud (free)
1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io>, click **New app**, and pick this repo and `cfb_dashboard.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   CFBD_API_KEY = "your-key"
   ```
4. Click **Deploy**.

## Notes
- Spreads are from the home team's view (negative = home favored). Check one game against a
  line you know to confirm the sign convention from the API.
- The model doesn't know about injuries, QB changes, or weather. Huge "edges" usually mean
  the model is missing something.
- Sidebar sliders (ridge shrinkage, margin/total std dev, last-season weight) are starting
  values, not tuned.
