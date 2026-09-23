"""
College Football Betting Dashboard (Streamlit)

Setup:
    pip install streamlit pandas numpy requests scikit-learn
    Get a free API key at https://collegefootballdata.com/key
    Save it in .streamlit/secrets.toml as:   CFBD_API_KEY = "your-key"
    (or just paste it into the sidebar when the app runs)

Run:
    streamlit run cfb_dashboard.py

How it works (short version):
    1. Pull finished games from the CollegeFootballData (CFBD) API.
    2. Fit a "points model": every team gets an offense number and a defense number.
       Predicted score = league average + team offense + opponent defense + home field.
    3. Turn the predicted scores into a spread, a total, and a win probability.
    4. Compare those to the market lines and compute expected value (EV) for each bet.
"""
import re
from datetime import date
from math import erf, sqrt

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import Ridge

API = "https://api.collegefootballdata.com"
OTHER = "FCS/Other"               # all non-FBS opponents get lumped into one "team"
STD_PROFIT = 100 / 110            # profit per $1 risked at standard -110 odds


# ---------------------------------------------------------------- small helpers
def norm_cdf(x):
    """Probability that a standard normal variable is below x."""
    return 0.5 * (1 + erf(x / sqrt(2)))


def ml_to_profit(ml):
    """American odds -> profit per $1 risked (+150 -> 1.5, -150 -> 0.667)."""
    return ml / 100 if ml > 0 else 100 / -ml


def field(d, name):
    """Read a key from an API dict whether it is camelCase or snake_case."""
    snake = re.sub(r"([A-Z])", lambda m: "_" + m.group(1).lower(), name)
    return d.get(name, d.get(snake))


# ---------------------------------------------------------------- data loading
@st.cache_data(ttl=3600, show_spinner="Fetching data from CFBD...")
def cfbd(endpoint, key, **params):
    """Call the CFBD API and return a DataFrame (cached for an hour)."""
    r = requests.get(f"{API}/{endpoint}", params=params, timeout=30,
                     headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    # the API can return snake_case or camelCase column names; standardize to camelCase
    df.columns = [re.sub(r"_([a-z])", lambda m: m.group(1).upper(), c) for c in df.columns]
    return df


def training_games(key, season, prior_weight):
    """Finished regular-season games from this season (weight 1) and last season (smaller weight)."""
    frames = []
    for year, weight in ((season, 1.0), (season - 1, prior_weight)):
        if weight == 0:
            continue
        g = cfbd("games", key, year=year, seasonType="regular")
        if g.empty:
            continue
        g = g.dropna(subset=["homePoints", "awayPoints"]).copy()
        g["weight"] = weight
        g["neutralSite"] = g["neutralSite"].fillna(False).astype(bool)
        # lump every non-FBS team into one bucket so we don't fit hundreds of tiny-sample teams
        for side in ("home", "away"):
            g.loc[g[f"{side}Classification"] != "fbs", f"{side}Team"] = OTHER
        frames.append(g[~((g["homeTeam"] == OTHER) & (g["awayTeam"] == OTHER))])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def market_lines(key, season, week):
    """Average spread/total (and median moneylines) across sportsbooks, one row per game."""
    raw = cfbd("lines", key, year=season, week=week, seasonType="regular")
    rows = []
    for _, g in raw.iterrows():
        for ln in g["lines"]:
            rows.append({"id": g["id"], "spread": field(ln, "spread"), "total": field(ln, "overUnder"),
                         "ml_home": field(ln, "homeMoneyline"), "ml_away": field(ln, "awayMoneyline")})
    if not rows:
        return pd.DataFrame(columns=["id", "spread", "total", "ml_home", "ml_away"])
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    return df.groupby("id", as_index=False).agg(spread="mean", total="mean", ml_home="median", ml_away="median")


# ---------------------------------------------------------------- the model
@st.cache_data
def fit_ratings(games, alpha):
    """
    Ridge regression on team scores.
    Each game makes two rows (one per team). Features = offense of the team + defense of the
    opponent + home field. Ridge shrinks ratings toward average so small samples don't go wild.
    """
    teams = sorted(set(games["homeTeam"]) | set(games["awayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    X = np.zeros((2 * len(games), 2 * T + 1))
    y, w = [], []
    r = 0
    for g in games.itertuples():
        hfa = 0 if g.neutralSite else 1
        for team, opp, pts, sign in ((g.homeTeam, g.awayTeam, g.homePoints, hfa),
                                     (g.awayTeam, g.homeTeam, g.awayPoints, -hfa)):
            X[r, idx[team]] = 1          # this team's offense
            X[r, T + idx[opp]] = 1       # opponent's defense (higher = allows more points)
            X[r, 2 * T] = sign           # +1 home, -1 away, 0 neutral
            y.append(pts)
            w.append(g.weight)
            r += 1
    model = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
    c = model.coef_
    return {"mu": model.intercept_, "hfa": c[2 * T],
            "off": dict(zip(teams, c[:T])), "dfn": dict(zip(teams, c[T:2 * T]))}


def predict(rt, home, away, neutral):
    """Predicted (home points, away points). Unknown teams (FCS, etc.) fall back to the OTHER bucket."""
    h = 0 if neutral else 1
    off = lambda t: rt["off"].get(t, rt["off"].get(OTHER, 0))
    dfn = lambda t: rt["dfn"].get(t, rt["dfn"].get(OTHER, 0))
    home_pts = rt["mu"] + off(home) + dfn(away) + rt["hfa"] * h
    away_pts = rt["mu"] + off(away) + dfn(home) - rt["hfa"] * h
    return home_pts, away_pts


def evaluate(home, away, neutral, spread, total, ml_home, ml_away, rt, s):
    """Model one game. Returns (summary row, list of bets). `spread` is from the HOME team's view."""
    hp, ap = predict(rt, home, away, neutral)
    margin, tot = hp - ap, hp + ap                       # margin > 0 means the home team wins
    p_home = norm_cdf(margin / s["sd_margin"])
    game = f"{away} @ {home}"
    bets = []

    def add(bet, p, profit, edge=np.nan):
        ev = p * profit - (1 - p)                        # expected profit per $1 risked
        kelly = max(ev / profit, 0)                      # Kelly fraction of bankroll
        stake = min(s["bankroll"] * s["kelly_frac"] * kelly, 0.05 * s["bankroll"])  # cap at 5%
        bets.append({"Game": game, "Bet": bet, "Model Win %": 100 * p, "Breakeven %": 100 / (1 + profit),
                     "Points Edge": edge, "EV %": 100 * ev, "Suggested Stake": stake})

    if not pd.isna(spread):
        edge = margin + spread                           # home covers if margin + spread > 0
        p_cover = norm_cdf(edge / s["sd_margin"])
        add(f"{home} {spread:+.1f}", p_cover, STD_PROFIT, edge)
        add(f"{away} {-spread:+.1f}", 1 - p_cover, STD_PROFIT, -edge)
    if not pd.isna(total):
        p_over = 1 - norm_cdf((total - tot) / s["sd_total"])
        add(f"Over {total:.1f}", p_over, STD_PROFIT, tot - total)
        add(f"Under {total:.1f}", 1 - p_over, STD_PROFIT, total - tot)
    if not pd.isna(ml_home) and not pd.isna(ml_away):
        add(f"{home} ML ({ml_home:+.0f})", p_home, ml_to_profit(ml_home))
        add(f"{away} ML ({ml_away:+.0f})", 1 - p_home, ml_to_profit(ml_away))

    summary = {"Game": game, "Model Score": f"{home} {hp:.0f} - {away} {ap:.0f}",
               "Model Spread": -margin, "Market Spread": spread, "Spread Edge": margin + spread,
               "Model Total": tot, "Market Total": total, "Total Edge": tot - total,
               "Home Win %": 100 * p_home}
    return summary, bets


# ---------------------------------------------------------------- the app
BET_COLS = {"Model Win %": st.column_config.NumberColumn(format="%.1f%%"),
            "Breakeven %": st.column_config.NumberColumn(format="%.1f%%"),
            "Points Edge": st.column_config.NumberColumn(format="%.1f"),
            "EV %": st.column_config.NumberColumn(format="%.1f%%"),
            "Suggested Stake": st.column_config.NumberColumn(format="$%.2f")}


def main():
    st.set_page_config(page_title="CFB Betting Dashboard", layout="wide")
    st.title("🏈 College Football Betting Dashboard")

    # ---- sidebar settings
    try:
        default_key = st.secrets["CFBD_API_KEY"]
    except Exception:
        default_key = ""
    today = date.today()
    guess_season = today.year if today.month >= 7 else today.year - 1
    guess_week = max(1, min(15, (today - date(guess_season, 8, 25)).days // 7))   # rough guess

    sb = st.sidebar
    key = sb.text_input("CFBD API key", value=default_key, type="password")
    season = sb.number_input("Season", 2015, 2100, guess_season)
    week = sb.number_input("Week to analyze", 1, 16, guess_week)
    sb.subheader("Model settings")
    prior_weight = sb.slider("Weight on last season's games", 0.0, 1.0, 0.4, 0.05,
                             help="0 = ignore last season. Useful early in the year when samples are small.")
    alpha = sb.slider("Ridge shrinkage", 1, 50, 10,
                      help="Higher = ratings pulled harder toward average.")
    s = {"sd_margin": sb.slider("Std dev of game margin (pts)", 10.0, 20.0, 15.0, 0.5),
         "sd_total": sb.slider("Std dev of game total (pts)", 8.0, 20.0, 13.0, 0.5)}
    sb.subheader("Bankroll")
    s["bankroll"] = sb.number_input("Bankroll ($)", 10, 1_000_000, 1000)
    s["kelly_frac"] = sb.slider("Kelly fraction", 0.0, 1.0, 0.25, 0.05,
                                help="Fraction of the full Kelly stake to use. 0.25 is a common, safer choice.")
    min_ev = sb.slider("Minimum EV % to show as a pick", 0.0, 15.0, 3.0, 0.5)

    if not key:
        st.info("Enter your free CFBD API key in the sidebar to get started (collegefootballdata.com/key).")
        st.stop()

    # ---- load data + fit model
    try:
        games = training_games(key, int(season), prior_weight)
        if games.empty:
            st.warning("No finished games found yet for that season. Try raising the last-season weight.")
            st.stop()
        rt = fit_ratings(games, alpha)
        sched = cfbd("games", key, year=int(season), week=int(week), seasonType="regular")
        mk = market_lines(key, int(season), int(week))
    except requests.HTTPError as e:
        st.error(f"CFBD API error: {e}. Check your API key.")
        st.stop()

    # ---- score every game on this week's slate
    slate, all_bets = [], []
    if not sched.empty:
        sched["neutralSite"] = sched["neutralSite"].fillna(False).astype(bool)
        sched = sched.merge(mk, on="id", how="left")
        for g in sched.itertuples():
            summary, bets = evaluate(g.homeTeam, g.awayTeam, g.neutralSite, g.spread, g.total,
                                     g.ml_home, g.ml_away, rt, s)
            slate.append(summary)
            all_bets += bets
    slate, all_bets = pd.DataFrame(slate), pd.DataFrame(all_bets)

    tab1, tab2, tab3, tab4 = st.tabs(["Best Bets", "Full Slate", "Matchup Explorer", "Team Ratings"])

    with tab1:
        st.caption(f"Week {week}: bets where the model's EV is at least {min_ev:.1f}%. "
                   "Spreads and totals assume -110 odds; moneylines use the median price across books.")
        if all_bets.empty:
            st.warning("No games or lines found for that week yet.")
        else:
            picks = all_bets[all_bets["EV %"] >= min_ev].sort_values("EV %", ascending=False)
            st.dataframe(picks, hide_index=True, column_config=BET_COLS)
            if (picks["Points Edge"].abs() > 7).any():
                st.warning("Some edges are 7+ points. That usually means something the model can't see "
                           "(injuries, a QB change, a tiny sample), not a free lunch. Double-check those.")

    with tab2:
        st.caption("Spread is from the home team's view (negative = home favored). "
                   "Edge = model minus market, so a positive Spread Edge favors the home team covering.")
        st.dataframe(slate, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.1f") for c in slate.columns
            if c not in ("Game", "Model Score")} if not slate.empty else None)

    with tab3:
        st.caption("Pick any two teams and type in the lines from your own sportsbook.")
        teams = [t for t in sorted(rt["off"]) if t != OTHER]
        c1, c2, c3 = st.columns(3)
        home = c1.selectbox("Home team", teams, index=0)
        away = c2.selectbox("Away team", teams, index=min(1, len(teams) - 1))
        neutral = c3.checkbox("Neutral site")
        c1, c2, c3, c4 = st.columns(4)
        spread = c1.number_input("Home spread", value=-3.5, step=0.5)
        total = c2.number_input("Over/under", value=55.5, step=0.5)
        ml_home = c3.number_input("Home moneyline", value=-150, step=5)
        ml_away = c4.number_input("Away moneyline", value=130, step=5)
        summary, bets = evaluate(home, away, neutral, spread, total, ml_home, ml_away, rt, s)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model score", summary["Model Score"])
        m2.metric("Model spread (home)", f"{summary['Model Spread']:+.1f}")
        m3.metric("Model total", f"{summary['Model Total']:.1f}")
        m4.metric("Home win %", f"{summary['Home Win %']:.1f}%")
        st.dataframe(pd.DataFrame(bets), hide_index=True, column_config=BET_COLS)

    with tab4:
        st.caption("Points vs. an average FBS team. Defense is flipped so higher = better for both columns.")
        ratings = pd.DataFrame({"Offense": rt["off"], "Defense": {t: -v for t, v in rt["dfn"].items()}})
        ratings["Overall"] = ratings["Offense"] + ratings["Defense"]
        ratings = ratings.drop(index=OTHER, errors="ignore").sort_values("Overall", ascending=False).round(2)
        st.bar_chart(ratings.head(25)["Overall"])
        st.dataframe(ratings)

    st.caption("For entertainment and learning. A simple ratings model rarely beats the closing line, "
               "so treat its 'edges' with skepticism and only risk money you can afford to lose.")


if __name__ == "__main__":
    main()
