"""
strategy.py
-----------
Pure strategy logic. NO broker keys, NO secrets, NO network calls here.
Safe to commit to GitHub as-is.

Takes OHLCV pandas DataFrames (from broker_upstox.py) and returns signals.
Combines everything built across prior sessions:
  - Regime gate (Nifty trend + ADX + VIX proxy)
  - Swing Setup A: RSI(2)/IBS pullback-to-trend (mean reversion)
  - Swing Setup B: Volatility Contraction Breakout (2-stage: watchlist + trigger)
  - Intraday Trend-Mode: VWAP + opening-range momentum
  - Intraday Range-Mode: VWAP mean-reversion

DataFrame contract expected by every function below:
    columns = ['open', 'high', 'low', 'close', 'volume']
    index   = pandas.DatetimeIndex, sorted ascending
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the indicators used by the swing setups to a DAILY OHLCV frame."""
    out = df.copy()
    out["sma50"] = SMAIndicator(out["close"], 50).sma_indicator()
    out["sma200"] = SMAIndicator(out["close"], 200).sma_indicator()
    out["adx14"] = ADXIndicator(out["high"], out["low"], out["close"], 14).adx()
    out["rsi2"] = RSIIndicator(out["close"], 2).rsi()
    out["atr14"] = AverageTrueRange(out["high"], out["low"], out["close"], 14).average_true_range()
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["vol_sma50"] = out["volume"].rolling(50).mean()
    out["range"] = out["high"] - out["low"]
    out["range_sma20"] = out["range"].rolling(20).mean()
    out["high_252"] = out["close"].rolling(252).max()
    out["high_50_prevbar"] = out["close"].rolling(50).max().shift(1)
    return out


def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds indicators used by the intraday setups to an INTRADAY OHLCV frame
    (expects one trading day's candles at a time, e.g. today's 15-min bars).
    """
    out = df.copy()
    out["ema20"] = EMAIndicator(out["close"], 20).ema_indicator()
    out["ema50"] = EMAIndicator(out["close"], 50).ema_indicator()
    out["adx14"] = ADXIndicator(out["high"], out["low"], out["close"], 14).adx()
    out["rsi14"] = RSIIndicator(out["close"], 14).rsi()
    out["atr14"] = AverageTrueRange(out["high"], out["low"], out["close"], 14).average_true_range()
    out["vol_sma20"] = out["volume"].rolling(20).mean()

    tp = (out["high"] + out["low"] + out["close"]) / 3
    day = out.index.date
    out["_tpv"] = tp * out["volume"]
    out["vwap"] = out.groupby(day)["_tpv"].cumsum() / out.groupby(day)["volume"].cumsum()
    out.drop(columns="_tpv", inplace=True)
    return out


def get_regime(nifty_daily: pd.DataFrame, india_vix_latest: float | None = None) -> str:
    """
    Returns one of: 'trending', 'range_bound', 'high_vol_avoid'.
    nifty_daily must already have indicators from add_daily_indicators().
    """
    row = nifty_daily.iloc[-1]

    if india_vix_latest is not None and india_vix_latest > 22:
        return "high_vol_avoid"
    if row["close"] < row["sma200"]:
        return "high_vol_avoid"
    if row["close"] > row["sma50"] > row["sma200"] and row["adx14"] > 18:
        return "trending"
    return "range_bound"


def swing_setup_a_pullback(df: pd.DataFrame, price_floor: float = 50) -> bool:
    """RSI(2)/IBS pullback-to-trend. df must have add_daily_indicators() applied."""
    r = df.iloc[-1]
    return bool(
        r["close"] > r["sma200"]
        and r["close"] > r["sma50"]
        and r["adx14"] > 20
        and r["rsi2"] < 10
        and r["volume"] < r["vol_sma20"]
        and r["close"] > price_floor
    )


def swing_setup_b_watchlist(df: pd.DataFrame, price_floor: float = 50) -> bool:
    """Volatility contraction — stage 1. Flags coiling candidates, NOT an entry signal."""
    r = df.iloc[-1]
    return bool(
        r["close"] > r["sma50"]
        and r["close"] > r["sma200"]
        and r["adx14"] > 20
        and r["close"] > 0.85 * r["high_252"]
        and r["range"] < 0.7 * r["range_sma20"]
        and r["volume"] < 0.65 * r["vol_sma50"]
        and r["close"] > price_floor
    )


def swing_setup_b_trigger(df: pd.DataFrame, price_floor: float = 50) -> bool:
    """Volatility contraction — stage 2. Run ONLY on names that passed the watchlist check."""
    r = df.iloc[-1]
    return bool(
        r["close"] > r["high_50_prevbar"]
        and r["volume"] > 2 * r["vol_sma50"]
        and r["adx14"] > 20
        and r["close"] > price_floor
    )


def evaluate_swing(df: pd.DataFrame, regime: str, price_floor: float = 50) -> dict:
    """Convenience wrapper: runs the regime-appropriate swing checks on one stock."""
    df = add_daily_indicators(df)
    if len(df) < 252:
        return {"setup_a": False, "setup_b_watchlist": False, "setup_b_trigger": False,
                "note": "Need 252+ daily bars for reliable signals"}
    result = {
        "setup_a": swing_setup_a_pullback(df, price_floor),
        "setup_b_watchlist": swing_setup_b_watchlist(df, price_floor),
        "setup_b_trigger": swing_setup_b_trigger(df, price_floor),
    }
    if regime == "high_vol_avoid":
        result = {k: False for k in result if k != "note"}
        result["note"] = "Regime = high_vol_avoid: no new swing entries"
    elif regime == "range_bound":
        result["setup_b_watchlist"] = False
        result["setup_b_trigger"] = False
        result["note"] = "Regime = range_bound: Setup A only, half size"
    else:
        result["note"] = "Regime = trending: both setups, full size"
    return result


def intraday_trend_mode(df: pd.DataFrame, price_floor: float = 50) -> dict:
    """VWAP + momentum, used when regime == 'trending'. df = today's 15-min bars so far."""
    df = add_intraday_indicators(df)
    if len(df) < 20:
        return {"buy": False, "sell": False, "note": "Not enough bars yet today"}
    r = df.iloc[-1]
    buy = bool(
        r["close"] > r["ema20"] > r["ema50"]
        and r["adx14"] > 25
        and 55 < r["rsi14"] < 72
        and r["close"] > r["vwap"]
        and r["volume"] > 2 * r["vol_sma20"]
        and r["close"] > price_floor
    )
    sell = bool(
        r["close"] < r["ema20"] < r["ema50"]
        and r["adx14"] > 25
        and 28 < r["rsi14"] < 45
        and r["close"] < r["vwap"]
        and r["volume"] > 2 * r["vol_sma20"]
        and r["close"] > price_floor
    )
    return {"buy": buy, "sell": sell}


def intraday_range_mode(df: pd.DataFrame, price_floor: float = 50) -> dict:
    """VWAP mean-reversion, used when regime == 'range_bound'."""
    df = add_intraday_indicators(df)
    if len(df) < 20:
        return {"fade_short": False, "note": "Not enough bars yet today"}
    r = df.iloc[-1]
    fade = bool(
        r["close"] < r["vwap"]
        and (r["vwap"] - r["close"]) > 0.5 * r["atr14"]
        and r["adx14"] < 20
        and r["rsi14"] < 35
        and r["volume"] > 1.5 * r["vol_sma20"]
        and r["close"] > price_floor
    )
    return {"fade_short": fade}


def evaluate_intraday(df: pd.DataFrame, regime: str, price_floor: float = 50) -> dict:
    """Convenience wrapper: routes to the right intraday setup based on regime."""
    if regime == "high_vol_avoid":
        return {"note": "Regime = high_vol_avoid: no new intraday entries"}
    if regime == "trending":
        return intraday_trend_mode(df, price_floor)
    return intraday_range_mode(df, price_floor)


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> int:
    """Returns number of shares to buy given a risk-per-trade rule. Never trade on this alone."""
    risk_amount = capital * (risk_pct / 100)
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        return 0
    return int(risk_amount // per_share_risk)


def swing_signal_label(result: dict) -> str:
    """result is the dict returned by evaluate_swing()."""
    if result.get("setup_a"):
        return "BUY (Setup A: Pullback)"
    if result.get("setup_b_trigger"):
        return "BUY (Setup B: Breakout)"
    if result.get("setup_b_watchlist"):
        return "WATCH (coiling — no entry yet)"
    return "NO SIGNAL"


def intraday_signal_label(result: dict) -> str:
    """result is the dict returned by evaluate_intraday()."""
    if result.get("buy"):
        return "BUY"
    if result.get("sell"):
        return "SELL"
    if result.get("fade_short"):
        return "SELL (VWAP fade)"
    return "NO SIGNAL"
