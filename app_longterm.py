"""
app_longterm.py
----------------
Long-Term (1M/3M/6M momentum) dashboard across Large/Mid/Small/Micro/Penny
cap tiers, plus a High-Momentum Swing (3-15 day) tab. Separate app from
app.py (the Intraday/Swing regime dashboard) — same login pattern, same
broker_upstox.py, so it reads the SAME Streamlit secrets.

READ THIS BEFORE TRUSTING THE OUTPUT:
Every tab ranks by price momentum + trend + liquidity — NOT fundamentals.
There is no reliable free live API for ROE/debt/promoter holding, so this
dashboard cannot check those automatically. Every row carries a manual
verification note for exactly this reason.
"""

import time
import streamlit as st
import pandas as pd
from datetime import date, timedelta

from broker_upstox import (get_login_url, exchange_code_for_token, get_historical_candles,
                            search_instrument_key)
from strategy import add_daily_indicators, get_regime
from longterm_strategy import evaluate_longterm, evaluate_high_momentum_swing
from universe import get_index_constituents, get_penny_candidate_symbols, CAP_TIER_INDEX

st.set_page_config(page_title="Long-Term & High-Momentum Swing — Upstox", layout="wide")
st.title("Long-Term (1M/3M/6M) & High-Momentum Swing Dashboard")

with st.sidebar:
    st.header("Upstox Login")
    api_key = st.secrets.get("UPSTOX_API_KEY", "") or st.text_input("API Key", type="password")
    api_secret = st.secrets.get("UPSTOX_API_SECRET", "") or st.text_input("API Secret", type="password")
    redirect_uri = st.secrets.get("UPSTOX_REDIRECT_URI", "") or st.text_input(
        "Redirect URI", value="https://your-app-name.streamlit.app")

    if "access_token" not in st.session_state:
        st.session_state.access_token = None

    query_params = st.query_params
    auth_code = query_params.get("code")

    if st.session_state.access_token:
        st.success("Logged in for this session.")
        if st.button("Log out"):
            st.session_state.access_token = None
            st.rerun()
    elif auth_code and api_key and api_secret and redirect_uri:
        try:
            st.session_state.access_token = exchange_code_for_token(api_key, api_secret, redirect_uri, auth_code)
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Token exchange failed: {e}")
    elif api_key and redirect_uri:
        login_url = get_login_url(api_key, redirect_uri)
        st.markdown(f"[Click here to log in to Upstox]({login_url})")
        st.caption("You'll be redirected back here automatically after logging in.")
    else:
        st.info("Enter API Key, API Secret, and Redirect URI to begin.")

if not st.session_state.access_token:
    st.stop()

token = st.session_state.access_token

if "lt_resolve_cache" not in st.session_state:
    st.session_state.lt_resolve_cache = {}
if "lt_index_cache" not in st.session_state:
    st.session_state.lt_index_cache = {}
if "lt_results" not in st.session_state:
    st.session_state.lt_results = {}


def resolve_symbols(symbols):
    resolved = []
    for symbol in symbols:
        if symbol in st.session_state.lt_resolve_cache:
            key = st.session_state.lt_resolve_cache[symbol]
        else:
            try:
                key = search_instrument_key(token, symbol)
            except Exception:
                key = None
            st.session_state.lt_resolve_cache[symbol] = key
            time.sleep(0.1)
        if key:
            resolved.append((symbol, key))
    return resolved


def get_cached_index(index_key):
    if index_key not in st.session_state.lt_index_cache:
        st.session_state.lt_index_cache[index_key] = get_index_constituents(index_key)
    return st.session_state.lt_index_cache[index_key]


NSE_LIMITATION_NOTE = (
    "If this fails, NSE is blocking the request from this server (common on cloud "
    "hosts) — this is an NSE-side block, not a bug here."
)


def run_longterm_scan(tier_name, price_floor, turnover_floor, price_ceiling=None):
    with st.spinner(f"Fetching {tier_name} universe..."):
        try:
            if tier_name == "Penny":
                symbols = get_penny_candidate_symbols()
            else:
                symbols = get_cached_index(CAP_TIER_INDEX[tier_name])
        except Exception as e:
            st.error(f"Could not fetch {tier_name} list from NSE: {e}\n\n{NSE_LIMITATION_NOTE}")
            return

    with st.spinner(f"Resolving {len(symbols)} instrument keys..."):
        resolved = resolve_symbols(symbols)

    today = date.today().isoformat()
    seven_months_ago = (date.today() - timedelta(days=220)).isoformat()

    rows = []
    progress = st.progress(0, text=f"Scanning {tier_name}...")
    for i, (name, key) in enumerate(resolved):
        try:
            df = get_historical_candles(token, key, "day", seven_months_ago, today)
            latest_close = df["close"].iloc[-1] if len(df) else None

            if tier_name == "Penny" and (latest_close is None or latest_close > price_ceiling):
                progress.progress((i + 1) / len(resolved))
                continue

            result = evaluate_longterm(df, price_floor, turnover_floor)
        except Exception as e:
            result = {"Verdict": "ERROR", "note": f"Error: {e}"}
        result["Stock"] = name
        rows.append(result)
        time.sleep(0.05)
        progress.progress((i + 1) / len(resolved))
    progress.empty()

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        verdict_order = {"STRONG (all horizons positive, above 50DMA)": 0, "WATCH (mostly positive, above 50DMA)": 1,
                          "MIXED": 2, "AVOID (below 50DMA)": 3, "SKIP (illiquid)": 4,
                          "INSUFFICIENT DATA": 5, "ERROR": 6}
        df_out["_sort"] = df_out["Verdict"].map(verdict_order).fillna(9)
        if "3M %" in df_out.columns:
            df_out = df_out.sort_values(["_sort", "3M %"], ascending=[True, False]).drop(columns="_sort")
        else:
            df_out = df_out.sort_values(["_sort"], ascending=[True]).drop(columns="_sort")
        cols = ["Stock", "Verdict", "1M %", "3M %", "6M %", "Trend", "% above 50DMA",
                "Liquidity OK", "note"]
        cols = [c for c in cols if c in df_out.columns]
        df_out = df_out[cols]
    st.session_state.lt_results[tier_name] = df_out


tab_large, tab_mid, tab_small, tab_micro, tab_penny, tab_swing = st.tabs(
    ["Large Cap", "Mid Cap", "Small Cap", "Micro Cap", "Penny", "Swing (3-15D High Momentum)"])

LONGTERM_HELP = (
    "Ranks by 1M/3M/6M price momentum, 50DMA trend, and liquidity — NOT fundamentals "
    "(no live free API for ROE/debt/promoter holding exists). Verify fundamentals on "
    "Screener.in before acting on anything shown here, especially Small/Micro/Penny."
)

for tier_name, tab in [("Large Cap", tab_large), ("Mid Cap", tab_mid), ("Small Cap", tab_small),
                       ("Micro Cap", tab_micro), ("Penny", tab_penny)]:
    with tab:
        st.caption(LONGTERM_HELP)
        c1, c2 = st.columns(2)
        price_floor = c1.number_input("Min price (Rs.)", value=10.0, key=f"pf_{tier_name}")
        turnover_floor = c2.number_input("Min daily turnover (Rs.)", value=10_000_000, key=f"tf_{tier_name}")
        price_ceiling = None
        if tier_name == "Penny":
            price_ceiling = st.number_input("Penny price ceiling (Rs.)", value=30.0, key="penny_ceiling")

        if st.button(f"Scan {tier_name}", key=f"scan_{tier_name}"):
            run_longterm_scan(tier_name, price_floor, turnover_floor, price_ceiling)

        if tier_name in st.session_state.lt_results:
            st.dataframe(st.session_state.lt_results[tier_name], hide_index=True, use_container_width=True)
        else:
            st.info(f"Click 'Scan {tier_name}' to run this tier.")

with tab_swing:
    st.caption(
        "3-15 day hold, high-momentum setup: EMA20>EMA50, ADX>25, RSI 60-75, "
        "1.5x volume, within 5% of its own 20-day high. Same fundamentals "
        "caveat applies — this is price/volume only."
    )
    swing_universe = st.multiselect(
        "Universe to scan", ["Large Cap", "Mid Cap", "Small Cap", "Micro Cap"],
        default=["Large Cap", "Mid Cap"], key="swing_universe")
    price_floor_sw = st.number_input("Min price (Rs.)", value=10.0, key="pf_swing")

    if st.button("Scan for High-Momentum Swing setups"):
        with st.spinner("Checking market regime..."):
            try:
                today = date.today().isoformat()
                year_ago = (date.today() - timedelta(days=400)).isoformat()
                nifty_daily = get_historical_candles(token, "NSE_INDEX|Nifty 50", "day", year_ago, today)
                nifty_daily = add_daily_indicators(nifty_daily)
                regime = get_regime(nifty_daily)
            except Exception as e:
                st.error(f"Could not fetch Nifty data: {e}")
                st.stop()
        st.metric("Market Regime", regime.replace("_", " ").title())
        if regime == "high_vol_avoid":
            st.warning("Regime is High-Vol-Avoid — momentum crash risk is elevated right now "
                       "(crowded winners are most vulnerable in volatile/reversing markets). "
                       "Results below are shown for reference; consider skipping new entries.")

        symbols = []
        for tier in swing_universe:
            try:
                symbols += get_cached_index(CAP_TIER_INDEX[tier])
            except Exception as e:
                st.error(f"Could not fetch {tier}: {e}")
        symbols = list(dict.fromkeys(symbols))

        resolved = resolve_symbols(symbols)
        today = date.today().isoformat()
        four_months_ago = (date.today() - timedelta(days=130)).isoformat()

        rows = []
        progress = st.progress(0, text="Scanning for high-momentum setups...")
        for i, (name, key) in enumerate(resolved):
            try:
                df = get_historical_candles(token, key, "day", four_months_ago, today)
                result = evaluate_high_momentum_swing(df, price_floor_sw)
            except Exception as e:
                result = {"Signal": "ERROR", "note": f"Error: {e}"}
            result["Stock"] = name
            rows.append(result)
            time.sleep(0.05)
            progress.progress((i + 1) / max(len(resolved), 1))
        progress.empty()

        df_swing = pd.DataFrame(rows)
        if not df_swing.empty:
            df_swing = df_swing[df_swing["Signal"] == "BUY (high momentum)"].reset_index(drop=True)
            cols = ["Stock", "Signal", "RSI", "ADX", "note"]
            cols = [c for c in cols if c in df_swing.columns]
            df_swing = df_swing[cols] if cols else df_swing
        st.session_state.swing_hm_results = df_swing

    if "swing_hm_results" in st.session_state:
        if st.session_state.swing_hm_results.empty:
            st.info("No high-momentum setups triggered right now.")
        else:
            st.dataframe(st.session_state.swing_hm_results, hide_index=True, use_container_width=True)
