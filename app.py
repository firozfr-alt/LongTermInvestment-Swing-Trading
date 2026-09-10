"""
app.py
------
Streamlit dashboard. Handles login/credentials and displays signals.
Secrets are read from Streamlit Cloud's Settings -> Secrets (see README) —
never hardcode your API key/secret in this file.
"""

import time
import streamlit as st
import pandas as pd
from datetime import date, timedelta

from broker_upstox import (get_login_url, exchange_code_for_token, get_historical_candles,
                            get_intraday_candles, search_instrument_key)
from strategy import (add_daily_indicators, get_regime, evaluate_swing, evaluate_intraday,
                       swing_signal_label, intraday_signal_label)
from universe import get_combined_universe

st.set_page_config(page_title="Intraday & Swing Strategy — Upstox", layout="wide")
st.title("Intraday & Swing Strategy Dashboard")

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

st.sidebar.header("Universe")
nifty_key = st.sidebar.text_input("Nifty instrument key (for regime check)", value="NSE_INDEX|Nifty 50")

use_nifty50 = st.sidebar.checkbox("Nifty 50", value=True)
use_next50 = st.sidebar.checkbox("Nifty Next 50", value=True)
use_banknifty = st.sidebar.checkbox("Bank Nifty", value=True)

extra_symbols_raw = st.sidebar.text_area(
    "Extra symbols to add (optional, one per line, NSE trading symbol)",
    value="", help="e.g. IRCTC — plain NSE trading symbols, no instrument_key needed anymore.")

price_floor = st.sidebar.number_input("Minimum price filter (Rs.)", value=50)

if "resolved_watchlist" not in st.session_state:
    st.session_state.resolved_watchlist = None
if "resolve_cache" not in st.session_state:
    st.session_state.resolve_cache = {}

if st.sidebar.button("Build watchlist"):
    index_keys = []
    if use_nifty50:
        index_keys.append("nifty50")
    if use_next50:
        index_keys.append("niftynext50")
    if use_banknifty:
        index_keys.append("banknifty")

    symbols = []
    if index_keys:
        try:
            symbols = get_combined_universe(index_keys)
        except Exception as e:
            st.sidebar.error(
                f"Could not fetch index list from NSE: {e}\n\n"
                "NSE sometimes blocks cloud servers — add symbols manually below instead."
            )
    for line in extra_symbols_raw.splitlines():
        s = line.strip().upper()
        if s and s not in symbols:
            symbols.append(s)

    resolved = []
    progress = st.sidebar.progress(0, text="Resolving instrument keys...")
    for i, symbol in enumerate(symbols):
        if symbol in st.session_state.resolve_cache:
            key = st.session_state.resolve_cache[symbol]
        else:
            try:
                key = search_instrument_key(token, symbol)
            except Exception:
                key = None
            st.session_state.resolve_cache[symbol] = key
            time.sleep(0.1)
        if key:
            resolved.append((symbol, key))
        progress.progress((i + 1) / max(len(symbols), 1))
    progress.empty()

    st.session_state.resolved_watchlist = resolved
    st.sidebar.success(f"Watchlist built: {len(resolved)} of {len(symbols)} symbols resolved.")

watchlist = st.session_state.resolved_watchlist or []
if watchlist:
    st.sidebar.caption(f"Current watchlist: {len(watchlist)} stocks")

if st.button("Run scan", disabled=not watchlist):
    today = date.today().isoformat()
    year_ago = (date.today() - timedelta(days=400)).isoformat()

    with st.spinner("Checking market regime..."):
        try:
            nifty_daily = get_historical_candles(token, nifty_key, "day", year_ago, today)
            nifty_daily = add_daily_indicators(nifty_daily)
            regime = get_regime(nifty_daily)
        except Exception as e:
            st.error(f"Could not fetch Nifty data: {e}")
            st.stop()

    st.session_state.regime = regime
    st.session_state.swing_rows = []
    st.session_state.intraday_rows = []

    scan_progress = st.progress(0, text="Scanning watchlist...")
    for i, (name, key) in enumerate(watchlist):
        try:
            df = get_historical_candles(token, key, "day", year_ago, today)
            result = evaluate_swing(df, regime, price_floor)
            result["Signal"] = swing_signal_label(result)
        except Exception as e:
            result = {"Signal": "ERROR", "note": f"Error: {e}"}
        result["Stock"] = name
        st.session_state.swing_rows.append(result)

        try:
            idf = get_intraday_candles(token, key, "30minute")
            iresult = evaluate_intraday(idf, regime, price_floor)
            iresult["Signal"] = intraday_signal_label(iresult)
        except Exception as e:
            iresult = {"Signal": "ERROR", "note": f"Error: {e}"}
        iresult["Stock"] = name
        st.session_state.intraday_rows.append(iresult)

        time.sleep(0.1)
        scan_progress.progress((i + 1) / len(watchlist))
    scan_progress.empty()

if "regime" in st.session_state:
    st.metric("Market Regime", st.session_state.regime.replace("_", " ").title())

    swing_tab, intraday_tab = st.tabs(["Swing", "Intraday"])

    with swing_tab:
        df = pd.DataFrame(st.session_state.swing_rows)
        cols = ["Stock", "Signal"] + [c for c in df.columns if c not in ("Stock", "Signal")]
        st.dataframe(df[cols], hide_index=True, use_container_width=True)

    with intraday_tab:
        df = pd.DataFrame(st.session_state.intraday_rows)
        cols = ["Stock", "Signal"] + [c for c in df.columns if c not in ("Stock", "Signal")]
        st.dataframe(df[cols], hide_index=True, use_container_width=True)

    st.caption(
        "Signals only — this app does not place orders automatically. "
        "Review each signal against your own risk rules before acting on it.")
else:
    st.info("In the sidebar: pick your indices, click 'Build watchlist', then click 'Run scan' above.")
