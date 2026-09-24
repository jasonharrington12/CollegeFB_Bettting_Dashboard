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
    3. Calibrate the model's uncertainty (sd_margin, sd_total) from real historical
       spread/total results so cover probabilities reflect actual hit rates.
    4. Turn the predicted scores into a spread, a total, and a win probability.
    5. Compare those to the market lines and compute expected value (EV) for each bet.
       Moneylines beyond ±400 are suppressed — extreme-dog MLs almost always look
       profitable on paper but never are in practice.
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


@st.cache_data(ttl=3600, show_spinner=False)
def team_conferences(key, season):
    """Return a dict of {team_name: conference} for FBS teams."""
    df = cfbd("teams/fbs", key, year=season)
    if df.empty or "conference" not in df.columns:
        return {}
    name_col = "school" if "school" in df.columns else df.columns[0]
    return dict(zip(df[name_col], df["conference"]))


@st.cache_data(ttl=3600, show_spinner=False)
def ap_rankings(key, season, week):
    """Return a set of team names that appear in the AP Top 25 for the given week."""
    df = cfbd("rankings", key, year=season, week=week, seasonType="regular")
    if df.empty:
        return set()
    # Each row has a 'polls' list; find the AP Top 25 poll
    ranked = set()
    for _, row in df.iterrows():
        for poll in row.get("polls", []):
            if poll.get("poll") == "AP Top 25":
                for entry in poll.get("ranks", []):
                    ranked.add(entry.get("school", ""))
    return ranked


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
    return df.groupby("id", as_index=False).agg(
        spread=("spread", "mean"),
        total=("total", "mean"),
        ml_home=("ml_home", "median"),
        ml_away=("ml_away", "median"),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def historical_lines(key, season):
    """
    Pull closing lines + results for completed games this season and last season.
    Returns a DataFrame with columns: model_margin, actual_margin, spread, total, actual_total
    used to calibrate sd_margin and sd_total from real data.
    """
    rows = []
    for year in (season, season - 1):
        games_df = cfbd("games", key, year=year, seasonType="regular")
        if games_df.empty:
            continue
        games_df = games_df.dropna(subset=["homePoints", "awayPoints"])
        lines_df = cfbd("lines", key, year=year, seasonType="regular")
        if lines_df.empty:
            continue
        # build a per-game closing spread (average across books)
        line_rows = []
        for _, g in lines_df.iterrows():
            for ln in g["lines"]:
                s = field(ln, "spread")
                ou = field(ln, "overUnder")
                if s is not None:
                    line_rows.append({"id": g["id"], "spread": s, "total": ou})
        if not line_rows:
            continue
        ldf = pd.DataFrame(line_rows).apply(pd.to_numeric, errors="coerce")
        ldf = ldf.groupby("id", as_index=False).agg(spread=("spread", "mean"), total=("total", "mean"))
        merged = games_df.merge(ldf, on="id", how="inner")
        for row in merged.itertuples():
            actual_margin = row.homePoints - row.awayPoints
            actual_total = row.homePoints + row.awayPoints
            rows.append({
                "actual_margin": actual_margin,
                "spread": row.spread,           # market spread (home team's view, negative = favored)
                "cover_error": actual_margin - (-row.spread),  # positive = home covered by more than expected
                "actual_total": actual_total,
                "total": row.total,
                "total_error": actual_total - row.total,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def calibrate_sigmas(hist_df, fallback_margin=15.0, fallback_total=13.0):
    """
    Estimate sd_margin and sd_total from how spread/total results are actually distributed.
    Uses the std dev of (actual_margin + market_spread) — i.e. how far games deviated from
    the closing line — which is the right sigma for cover-probability calculations.
    Falls back to provided defaults if there isn't enough data.
    """
    sd_margin = fallback_margin
    sd_total = fallback_total
    if hist_df.empty:
        return sd_margin, sd_total
    ce = hist_df["cover_error"].dropna()
    te = hist_df["total_error"].dropna()
    if len(ce) >= 20:
        sd_margin = float(ce.std())
    if len(te) >= 20:
        sd_total = float(te.std())
    return sd_margin, sd_total


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


# ML bets beyond this threshold are too extreme to be reliable — suppress them.
ML_ODDS_CAP = 400

# Alternate spread offsets (applied on top of the market spread)
ALT_SPREAD_OFFSETS = [-6.5, -3.5, 3.5, 6.5]


def parlay_american_odds(win_probs):
    """
    Given a list of true win probabilities, compute the fair parlay payout (American odds)
    and expected value vs the standard -110 parlay pricing sportsbooks use.
    Returns (fair_odds, book_payout, ev_pct).
    """
    if not win_probs:
        return None, None, None
    combined_p = 1.0
    for p in win_probs:
        combined_p *= p
    # Fair American odds for the parlay winner
    fair_profit = (1 / combined_p) - 1
    fair_odds = fair_profit * 100 if fair_profit <= 1 else fair_profit * 100

    # Standard book parlay payout: each leg priced at -110
    leg_decimal = 1 + STD_PROFIT          # 1.909...
    book_decimal = leg_decimal ** len(win_probs)
    book_profit = book_decimal - 1
    book_odds = int(book_profit * 100) if book_profit <= 1 else int(book_profit * 100)

    ev_pct = 100 * (combined_p * book_profit - (1 - combined_p))
    return combined_p, book_odds, ev_pct


def evaluate(home, away, neutral, spread, total, ml_home, ml_away, rt, s, alt_spreads=False):
    """Model one game. Returns (summary row, list of bets). `spread` is from the HOME team's view."""
    hp, ap = predict(rt, home, away, neutral)
    margin, tot = hp - ap, hp + ap                       # margin > 0 means the home team wins
    p_home = norm_cdf(margin / s["sd_margin"])
    game = f"{away} @ {home}"
    bets = []

    def add(bet, p, profit, edge=np.nan, bet_type="Spread"):
        ev = p * profit - (1 - p)                        # expected profit per $1 risked
        kelly = max(ev / profit, 0)                      # Kelly fraction of bankroll
        stake = min(s["bankroll"] * s["kelly_frac"] * kelly, 0.05 * s["bankroll"])  # cap at 5%
        bets.append({"Game": game, "Bet": bet, "Type": bet_type,
                     "Model Win %": 100 * p, "Breakeven %": 100 / (1 + profit),
                     "Points Edge": edge, "EV %": 100 * ev, "Suggested Stake": stake})

    if not pd.isna(spread):
        edge = margin + spread                           # home covers if margin + spread > 0
        p_cover = norm_cdf(edge / s["sd_margin"])
        add(f"{home} {spread:+.1f}", p_cover, STD_PROFIT, edge, "Spread")
        add(f"{away} {-spread:+.1f}", 1 - p_cover, STD_PROFIT, -edge, "Spread")
        # alternate spreads: shift the market line by each offset
        if alt_spreads:
            for offset in ALT_SPREAD_OFFSETS:
                alt = spread + offset                    # e.g. market -7 + 3.5 = alt -3.5
                alt_edge = margin + alt
                p_alt = norm_cdf(alt_edge / s["sd_margin"])
                add(f"{home} {alt:+.1f} (alt)", p_alt, STD_PROFIT, alt_edge, "Alt Spread")
                add(f"{away} {-alt:+.1f} (alt)", 1 - p_alt, STD_PROFIT, -alt_edge, "Alt Spread")
    if not pd.isna(total):
        p_over = 1 - norm_cdf((total - tot) / s["sd_total"])
        add(f"Over {total:.1f}", p_over, STD_PROFIT, tot - total, "Total")
        add(f"Under {total:.1f}", 1 - p_over, STD_PROFIT, total - tot, "Total")
    # Only show ML bets when the odds are within a realistic range (±400).
    # Extreme underdog MLs (+500, +1000, etc.) produce misleadingly high model EV
    # because a simple ratings model can't reliably price 10-to-1 shots.
    if (not pd.isna(ml_home) and not pd.isna(ml_away)
            and abs(ml_home) <= ML_ODDS_CAP and abs(ml_away) <= ML_ODDS_CAP):
        add(f"{home} ML ({ml_home:+.0f})", p_home, ml_to_profit(ml_home), bet_type="ML")
        add(f"{away} ML ({ml_away:+.0f})", 1 - p_home, ml_to_profit(ml_away), bet_type="ML")

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
        conf_map = team_conferences(key, int(season))
        ranked_teams = ap_rankings(key, int(season), int(week))
        hist_df = historical_lines(key, int(season))
    except requests.HTTPError as e:
        st.error(f"CFBD API error: {e}. Check your API key.")
        st.stop()

    # ---- calibrate sigmas from historical spread/total results
    cal_sd_margin, cal_sd_total = calibrate_sigmas(
        hist_df,
        fallback_margin=s["sd_margin"],
        fallback_total=s["sd_total"],
    )
    # Override the sidebar manual values with data-calibrated ones, but let the sidebar
    # sliders serve as a floor so the user can still nudge them if desired.
    s["sd_margin"] = max(s["sd_margin"], cal_sd_margin)
    s["sd_total"] = max(s["sd_total"], cal_sd_total)

    n_hist = len(hist_df) if not hist_df.empty else 0
    if n_hist >= 20:
        sb.caption(f"📊 Sigmas auto-calibrated from {n_hist} historical games: "
                   f"margin σ={cal_sd_margin:.1f} pts, total σ={cal_sd_total:.1f} pts.")
    else:
        sb.caption("⚠️ Not enough historical lines data to auto-calibrate — using manual slider values.")

    # ---- sidebar filters (conference + Top 25)
    sb.subheader("Filters")
    all_confs = sorted({c for c in conf_map.values() if c})
    sel_confs = sb.multiselect("Conference", all_confs, default=[],
                               help="Show only games where at least one team is in the selected conference(s). "
                                    "Leave blank for all games.")
    top25_only = sb.checkbox("Top 25 games only",
                             help="Show only games where at least one team is AP-ranked this week.")

    # ---- score every game on this week's slate
    slate, all_bets = [], []
    if not sched.empty:
        sched["neutralSite"] = sched["neutralSite"].fillna(False).astype(bool)
        sched = sched.merge(mk, on="id", how="left")
        for g in sched.itertuples():
            home_conf = conf_map.get(g.homeTeam, "")
            away_conf = conf_map.get(g.awayTeam, "")
            # conference filter
            if sel_confs and home_conf not in sel_confs and away_conf not in sel_confs:
                continue
            # Top 25 filter
            if top25_only and g.homeTeam not in ranked_teams and g.awayTeam not in ranked_teams:
                continue
            summary, bets = evaluate(g.homeTeam, g.awayTeam, g.neutralSite, g.spread, g.total,
                                     g.ml_home, g.ml_away, rt, s)
            # tag the summary row with conference info
            summary["Home Conf"] = home_conf
            summary["Away Conf"] = away_conf
            slate.append(summary)
            all_bets += bets
    slate, all_bets = pd.DataFrame(slate), pd.DataFrame(all_bets)

    # ---- re-score slate with alt spreads enabled (used for Best Bets + Parlay tabs)
    slate_alt, all_bets_alt = [], []
    if not sched.empty:
        for g in sched.itertuples():
            home_conf = conf_map.get(g.homeTeam, "")
            away_conf = conf_map.get(g.awayTeam, "")
            if sel_confs and home_conf not in sel_confs and away_conf not in sel_confs:
                continue
            if top25_only and g.homeTeam not in ranked_teams and g.awayTeam not in ranked_teams:
                continue
            summary, bets = evaluate(g.homeTeam, g.awayTeam, g.neutralSite, g.spread, g.total,
                                     g.ml_home, g.ml_away, rt, s, alt_spreads=True)
            summary["Home Conf"] = home_conf
            summary["Away Conf"] = away_conf
            slate_alt.append(summary)
            all_bets_alt += bets
    all_bets_alt = pd.DataFrame(all_bets_alt)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Best Bets", "Full Slate", "Matchup Explorer", "Team Ratings", "Parlay Builder"])

    with tab1:
        st.caption(f"Week {week}: bets where the model's EV is at least {min_ev:.1f}%. "
                   "Spreads and totals assume -110 odds; moneylines use the median price across books.")
        if all_bets_alt.empty:
            st.warning("No games or lines found for that week yet.")
        else:
            # ---- bet type filter
            all_types = ["Spread", "Alt Spread", "Total", "ML"]
            sel_types = st.multiselect("Bet type", all_types, default=["Spread", "Total", "ML"],
                                       key="tab1_type_filter",
                                       help="Filter by bet type. 'Alt Spread' shows ±3.5 / ±6.5 alternate lines.")
            picks = all_bets_alt[all_bets_alt["EV %"] >= min_ev].copy()
            if sel_types:
                picks = picks[picks["Type"].isin(sel_types)]
            picks = picks.sort_values("EV %", ascending=False)
            # join conference info from slate so we can filter by it
            if not slate.empty:
                picks = picks.merge(slate[["Game", "Home Conf", "Away Conf"]].drop_duplicates(),
                                    on="Game", how="left")
                tab1_confs = sorted({c for c in slate["Home Conf"].tolist() + slate["Away Conf"].tolist() if c})
                sel_tab1_confs = st.multiselect("Filter by conference", tab1_confs, default=[],
                                                key="tab1_conf_filter",
                                                help="Show only bets where at least one team is in the selected conference(s).")
                if sel_tab1_confs:
                    picks = picks[picks["Home Conf"].isin(sel_tab1_confs) | picks["Away Conf"].isin(sel_tab1_confs)]
                picks = picks.drop(columns=["Home Conf", "Away Conf"], errors="ignore")
            BET_COLS_WITH_TYPE = {**BET_COLS}
            st.dataframe(picks, hide_index=True, column_config=BET_COLS_WITH_TYPE)
            if not picks.empty and (picks["Points Edge"].abs() > 7).any():
                st.warning("Some edges are 7+ points. That usually means something the model can't see "
                           "(injuries, a QB change, a tiny sample), not a free lunch. Double-check those.")

    with tab2:
        st.caption("Spread is from the home team's view (negative = home favored). "
                   "Edge = model minus market, so a positive Spread Edge favors the home team covering.")
        if not slate.empty:
            tab2_confs = sorted({c for c in slate["Home Conf"].tolist() + slate["Away Conf"].tolist() if c})
            sel_tab2_confs = st.multiselect("Filter by conference", tab2_confs, default=[],
                                            key="tab2_conf_filter",
                                            help="Filter rows to games where at least one team is in the selected conference(s).")
            filtered_slate = slate
            if sel_tab2_confs:
                mask = slate["Home Conf"].isin(sel_tab2_confs) | slate["Away Conf"].isin(sel_tab2_confs)
                filtered_slate = slate[mask]
            st.dataframe(filtered_slate, hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="%.1f") for c in filtered_slate.columns
                if c not in ("Game", "Model Score", "Home Conf", "Away Conf")})
        else:
            st.warning("No games found for that week yet.")

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

    with tab5:
        st.caption("Pick your legs below. The builder shows combined win probability, "
                   "estimated payout at standard -110 parlay pricing, and overall EV.")
        if all_bets_alt.empty:
            st.warning("No bets available — check your season/week settings.")
        else:
            # Only offer positive-EV bets as parlay legs to keep the list useful
            parlay_pool = all_bets_alt[all_bets_alt["EV %"] >= min_ev].copy()
            # Let user also filter by type for the parlay pool
            p_types = st.multiselect("Include bet types in parlay pool", ["Spread", "Alt Spread", "Total", "ML"],
                                     default=["Spread", "Total", "ML"], key="parlay_type_filter")
            if p_types:
                parlay_pool = parlay_pool[parlay_pool["Type"].isin(p_types)]
            parlay_pool = parlay_pool.sort_values("EV %", ascending=False)

            if parlay_pool.empty:
                st.info("No positive-EV bets match the current filters. Lower the minimum EV slider or change bet types.")
            else:
                leg_options = parlay_pool["Bet"].tolist()
                # Suggest the top 3 by EV as a default starting point
                default_legs = leg_options[:min(3, len(leg_options))]
                chosen_legs = st.multiselect("Select parlay legs", leg_options, default=default_legs,
                                             help="Choose 2–8 legs. The builder assumes each leg is independent.")

                if len(chosen_legs) < 2:
                    st.info("Select at least 2 legs to build a parlay.")
                else:
                    chosen = parlay_pool[parlay_pool["Bet"].isin(chosen_legs)].drop_duplicates("Bet")
                    win_probs = (chosen["Model Win %"] / 100).tolist()
                    combined_p, book_odds, ev_pct = parlay_american_odds(win_probs)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Legs", len(chosen_legs))
                    m2.metric("Combined Win %", f"{100 * combined_p:.2f}%")
                    book_odds_str = f"+{book_odds}" if book_odds >= 0 else str(book_odds)
                    m3.metric("Est. Payout (book)", book_odds_str)
                    ev_color = "normal" if ev_pct >= 0 else "inverse"
                    m4.metric("Parlay EV %", f"{ev_pct:.1f}%", delta=f"{ev_pct:.1f}%", delta_color=ev_color)

                    st.subheader("Parlay legs")
                    st.dataframe(chosen[["Game", "Bet", "Type", "Model Win %", "EV %", "Points Edge"]],
                                 hide_index=True, column_config=BET_COLS)

                    if len(chosen_legs) > 4:
                        st.warning("Parlays of 5+ legs have very low hit rates even with positive EV. "
                                   "Consider splitting into smaller parlays.")

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
