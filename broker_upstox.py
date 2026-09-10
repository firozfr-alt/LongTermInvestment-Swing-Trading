"""
broker_upstox.py
-----------------
Thin wrapper around the Upstox v2 API. NO secrets are hardcoded anywhere in
this file — api_key, api_secret, redirect_uri, and access_token are always
passed in at call time by app.py, which reads them from Streamlit secrets.
Safe to commit to GitHub as-is.
"""

import requests
import pandas as pd

BASE_URL = "https://api.upstox.com/v2"


def get_login_url(api_key: str, redirect_uri: str) -> str:
    return (
        f"{BASE_URL}/login/authorization/dialog"
        f"?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}"
    )


def exchange_code_for_token(api_key: str, api_secret: str, redirect_uri: str, code: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/login/authorization/token",
        headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        data={
            "code": code,
            "client_id": api_key,
            "client_secret": api_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}


def get_historical_candles(access_token: str, instrument_key: str, interval: str,
                            from_date: str, to_date: str) -> pd.DataFrame:
    url = f"{BASE_URL}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
    resp = requests.get(url, headers=_headers(access_token), timeout=15)
    resp.raise_for_status()
    candles = resp.json()["data"]["candles"]
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def get_intraday_candles(access_token: str, instrument_key: str, interval: str = "30minute") -> pd.DataFrame:
    url = f"{BASE_URL}/historical-candle/intraday/{instrument_key}/{interval}"
    resp = requests.get(url, headers=_headers(access_token), timeout=15)
    resp.raise_for_status()
    candles = resp.json()["data"]["candles"]
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def get_ltp(access_token: str, instrument_keys: list[str]) -> dict:
    url = f"{BASE_URL}/market-quote/ltp"
    resp = requests.get(url, headers=_headers(access_token),
                         params={"instrument_key": ",".join(instrument_keys)}, timeout=15)
    resp.raise_for_status()
    return resp.json()["data"]


def search_instrument_key(access_token: str, trading_symbol: str) -> str | None:
    """
    Looks up the Upstox instrument_key for an NSE equity trading symbol
    (e.g. 'RELIANCE' -> 'NSE_EQ|INE002A01018') using Upstox's Instrument
    Search API, so you never have to hand-copy instrument_keys yourself.
    Returns None if nothing matches.
    """
    url = f"{BASE_URL}/instruments/search"
    resp = requests.get(
        url, headers=_headers(access_token),
        params={"query": trading_symbol, "exchanges": "NSE", "segments": "EQ",
                "instrument_types": "EQ", "records": 5},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("data", [])
    for item in results:
        symbol = item.get("trading_symbol") or item.get("symbol") or ""
        if symbol.upper() == trading_symbol.upper():
            return item.get("instrument_key")
    if results:
        return results[0].get("instrument_key")
    return None


def place_order(access_token: str, instrument_key: str, quantity: int, side: str,
                 order_type: str = "MARKET", product: str = "I", price: float = 0,
                 confirm_live: bool = False) -> dict:
    payload = {
        "quantity": quantity,
        "product": product,
        "validity": "DAY",
        "price": price,
        "instrument_token": instrument_key,
        "order_type": order_type,
        "transaction_type": side,
        "disclosed_quantity": 0,
        "trigger_price": 0,
        "is_amo": False,
    }
    if not confirm_live:
        return {"dry_run": True, "would_send": payload}

    resp = requests.post(f"{BASE_URL}/order/place", headers=_headers(access_token), json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()
