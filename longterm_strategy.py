"""
longterm_strategy.py
---------------------
Pure logic for the Long-Term (1M/3M/6M) and High-Momentum Swing (3-15 day)
dashboards. No secrets, no network calls. Safe to commit to GitHub.

IMPORTANT LIMITATION (read this before trusting the output):
This ranks stocks by PRICE MOMENTUM, TREND, and LIQUIDITY only — all of
which Upstox's price/volume API can actually verify live. It does NOT check
fundamentals (ROE, debt, promoter holding, pledge %) because there is no
reliable free live API for that data. A stock can rank #1 here purely on
price action while having weak or even fraudulent fundamentals underneath —
this is a documented, specific risk in penny/micro-cap stocks (inflated
press releases, paid stock promotion, concentrated promoter ownership used
to manipulate float). Treat every result here as a shortlist to manually
verify on Screener.in (promoter holding trend, pledge %, debt, red flags in
recent filings) before acting — not as a final buy signal, especially for
the Small/Micro/Penny tiers.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator

TRADING_DAYS = {"1M": 21, "3M": 63, "6M": 126}


def add_longterm_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df = daily OHLCV, at least ~130 trading days for the 6M lookback to work."""
    out = df.copy()
    out["sma50"] = SMAIndicator(out["close"], 50).sma_indicator()
    if len(out) > 60:
        out["sma200"] = SMAIndicator(out["close"], min(200, len(out) - 1)).sma_indicator()
    else:
        out["sma200"] = np.nan
    out["adx14"] = ADXIndicator(out["high"], out["low"], out["close"], 14).adx()
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    return out


def horizon_return(df: pd.DataFrame, trading_days: int):
    """% price return over the given number of trading days. None if not enough history."""
    if len(df) <= trading_days:
        return None
    start_price = df["close"].iloc[-trading_days - 1]
    end_price = df["close"].iloc[-1]
    if start_price <= 0:
        return None
    return round((end_price / start_price - 1) * 100, 2)


def evaluate_longterm(df: pd.DataFrame, price_floor: float = 10, turnover_floor: float = 10_000_000) -> dict:
    """
    Returns 1M/3M/6M returns, a trend flag, a liquidity flag, and a plain
    Verdict. df must be daily OHLCV with enough history for the horizons
    you care about (6M needs ~130+ trading days; fewer is fine, that
    horizon just won't be computable for a recently-listed stock).
    """
    if len(df) < 25:
        return {"Verdict": "INSUFFICIENT DATA", "1M %": None, "3M %": None, "6M %": None,
                "Trend": None, "note": "Fewer than ~25 trading days of history"}

    ind = add_longterm_indicators(df)
    latest = ind.iloc[-1]

    result = {
        "1M %": horizon_return(df, TRADING_DAYS["1M"]),
        "3M %": horizon_return(df, TRADING_DAYS["3M"]),
        "6M %": horizon_return(df, TRADING_DAYS["6M"]),
    }

    liquidity_ok = bool(latest["close"] > price_floor and latest["close"] * latest["volume"] > turnover_floor)
    trend_ok = bool(latest["close"] > latest["sma50"]) if not pd.isna(latest["sma50"]) else None

    result["Trend"] = "Above 50DMA" if trend_ok else ("Below 50DMA" if trend_ok is False else "N/A")
    result["Liquidity OK"] = liquidity_ok

    returns_available = [v for v in (result["1M %"], result["3M %"], result["6M %"]) if v is not None]
    positive_count = sum(1 for v in returns_available if v > 0)

    if not liquidity_ok:
        result["Verdict"] = "SKIP (illiquid)"
    elif trend_ok and returns_available and positive_count == len(returns_available):
        result["Verdict"] = "STRONG (all horizons positive, above 50DMA)"
    elif trend_ok and returns_available and positive_count >= max(1, len(returns_available) - 1):
        result["Verdict"] = "WATCH (mostly positive, above 50DMA)"
    elif trend_ok is False:
        result["Verdict"] = "AVOID (below 50DMA)"
    else:
        result["Verdict"] = "MIXED"

    # Flag overextension: a stock trading far above its own 50DMA is the
    # classic shape of a recent pump, regardless of which horizon you look
    # at (a recent spike inflates 1M/3M/6M returns alike, since they all
    # end "today" — so comparing horizons against each other doesn't catch
    # it; distance from the 50DMA does).
    pct_above_50dma = None
    if not pd.isna(latest["sma50"]) and latest["sma50"] > 0:
        pct_above_50dma = round((latest["close"] / latest["sma50"] - 1) * 100, 1)
    result["% above 50DMA"] = pct_above_50dma

    if pct_above_50dma is not None and pct_above_50dma > 35:
        result["note"] = (f"Trading {pct_above_50dma}% above its 50DMA — extended/possible pump shape, "
                           "verify manually before chasing, don't buy the spike")
    else:
        result["note"] = "Verify promoter holding/pledge/debt on Screener.in before acting"

    return result


def evaluate_high_momentum_swing(df: pd.DataFrame, price_floor: float = 10) -> dict:
    """
    3-15 day hold, high-momentum swing setup: EMA20 > EMA50 stack (200DMA
    intentionally not required — many small/micro stocks don't have 200
    days of history), ADX>25 trend strength, RSI 60-75 momentum band
    (avoids both weak momentum and already-overextended >75 readings),
    volume >1.5x average, price within 5% of its own 20-day high.
    """
    if len(df) < 25:
        return {"Signal": "INSUFFICIENT DATA"}

    out = df.copy()
    out["ema20"] = EMAIndicator(out["close"], 20).ema_indicator()
    out["ema50"] = EMAIndicator(out["close"], 50).ema_indicator()
    out["adx14"] = ADXIndicator(out["high"], out["low"], out["close"], 14).adx()
    out["rsi14"] = RSIIndicator(out["close"], 14).rsi()
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["high_20"] = out["close"].rolling(20).max()

    r = out.iloc[-1]
    if pd.isna(r["ema50"]) or pd.isna(r["adx14"]):
        return {"Signal": "INSUFFICIENT DATA"}

    triggered = bool(
        r["close"] > r["ema20"] > r["ema50"]
        and r["adx14"] > 25
        and 60 < r["rsi14"] < 75
        and r["volume"] > 1.5 * r["vol_sma20"]
        and r["close"] >= 0.95 * r["high_20"]
        and r["close"] > price_floor
    )
    return {
        "Signal": "BUY (high momentum)" if triggered else "NO SIGNAL",
        "RSI": round(r["rsi14"], 1) if not pd.isna(r["rsi14"]) else None,
        "ADX": round(r["adx14"], 1) if not pd.isna(r["adx14"]) else None,
        "note": "Hold 3-15 days; hard time-exit on day 15 if no target/stop hit" if triggered else "",
    }
