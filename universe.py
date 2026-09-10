""
universe.py
-----------
Fetches index constituent lists LIVE from NSE's own published CSV files.
"""

import requests
import pandas as pd
from io import StringIO

INDEX_CSV_URLS = {
    "nifty50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "niftynext50": "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "banknifty": "https://nsearchives.nseindia.com/content/indices/ind_niftybanklist.csv",
    "nifty100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "niftymidcap150": "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "niftysmallcap250": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "niftymicrocap250": "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
}


def _nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    })
    session.get("https://www.nseindia.com/", timeout=10)
    return session


def get_index_constituents(index_key: str) -> list[str]:
    if index_key not in INDEX_CSV_URLS:
        raise ValueError(f"Unknown index_key: {index_key}")
    session = _nse_session()
    resp = session.get(INDEX_CSV_URLS[index_key], timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    return df["Symbol"].astype(str).str.strip().tolist()


def get_combined_universe(index_keys: list[str]) -> list[str]:
    seen = set()
    combined = []
    for key in index_keys:
        for symbol in get_index_constituents(key):
            if symbol not in seen:
                seen.add(symbol)
                combined.append(symbol)
    return combined


CAP_TIER_INDEX = {
    "Large Cap": "nifty100",
    "Mid Cap": "niftymidcap150",
    "Small Cap": "niftysmallcap250",
    "Micro Cap": "niftymicrocap250",
}


def get_penny_candidate_symbols() -> list[str]:
    return get_combined_universe(["niftysmallcap250", "niftymicrocap250"])
