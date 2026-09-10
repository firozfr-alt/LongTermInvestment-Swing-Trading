"""
universe.py
-----------
Fetches index constituent lists LIVE from NSE's own published CSV files,
instead of hardcoding a stock list here.

Why not hardcode? Index constituents change every 6 months (NSE
reconstitution). A hardcoded list would silently go stale. Fetching live
keeps it always correct.

Known limitation: NSE's site sometimes blocks requests that don't look like
a real browser, and occasionally blocks cloud-hosted IPs (including
Streamlit Community Cloud) outright. If get_index_constituents() raises an
error, that's NSE blocking the request, not a bug in this code — fall back
to typing symbols manually in that case.
"""

import requests
import pandas as pd
from io import StringIO

INDEX_CSV_URLS = {
    "nifty50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "niftynext50": "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "banknifty": "https://nsearchives.nseindia.com/content/indices/ind_niftybanklist.csv",
    # Cap-tier universes for the Long-Term dashboard — mapped to NSE's own
    # official index lists rather than AMFI's PDF-only classification (AMFI
    # doesn't publish a clean machine-readable file; these NSE CSVs do).
    "nifty100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",          # Large Cap
    "niftymidcap150": "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",   # Mid Cap
    "niftysmallcap250": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",  # Small Cap
    "niftymicrocap250": "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250list.csv",  # Micro Cap
}


def _nse_session() -> requests.Session:
    """NSE requires a browser-like session (User-Agent + a prior visit to the
    homepage to pick up cookies) or it returns 403 on the CSV endpoints."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    })
    session.get("https://www.nseindia.com/", timeout=10)  # sets cookies
    return session


def get_index_constituents(index_key: str) -> list[str]:
    """
    index_key: one of the keys in INDEX_CSV_URLS above.
    Returns a list of NSE trading symbols, e.g. ['RELIANCE', 'TCS', ...].
    """
    if index_key not in INDEX_CSV_URLS:
        raise ValueError(f"Unknown index_key: {index_key}")
    session = _nse_session()
    resp = session.get(INDEX_CSV_URLS[index_key], timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    return df["Symbol"].astype(str).str.strip().tolist()


def get_combined_universe(index_keys: list[str]) -> list[str]:
    """Fetches multiple indices and returns a deduplicated symbol list, order preserved."""
    seen = set()
    combined = []
    for key in index_keys:
        for symbol in get_index_constituents(key):
            if symbol not in seen:
                seen.add(symbol)
                combined.append(symbol)
    return combined


# Human labels for the cap-tier keys used by the Long-Term dashboard
CAP_TIER_INDEX = {
    "Large Cap": "nifty100",
    "Mid Cap": "niftymidcap150",
    "Small Cap": "niftysmallcap250",
    "Micro Cap": "niftymicrocap250",
}


def get_penny_candidate_symbols() -> list[str]:
    """
    'Penny stock' has no official market-cap definition — it's a price-level
    term, not a cap tier. Real penny stocks live inside Small Cap + Micro Cap,
    so this pulls that combined pool; the app then keeps only the names
    actually trading at/below your price ceiling once their close price is
    fetched (no separate LTP call needed — the daily candle fetch already
    gives the latest close).
    """
    return get_combined_universe(["niftysmallcap250", "niftymicrocap250"])
