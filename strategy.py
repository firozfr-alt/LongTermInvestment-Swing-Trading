"""
strategy.py
-----------
Production-grade Swing & Intraday strategy logic.
Features:
- Market Regime detection via Nifty 50 / VIX
- Setup A: Quantitative Mean-Reversion Pullback with turnaround confirmation
- Setup B: High-Momentum Volatility Contraction Pattern (VCP) Breakout
- Automated Trade Plan Engine (Entry, Stop Loss, Target 1, Target 2, R:R)
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    # Trend Baselines
    out["sma50"] = SMAIndicator(out["close"], 50).sma_indicator()
    out["sma200"] = SMAIndicator(out["close"], 200).sma_indicator()
    out["ema20"] = EMAIndicator(out["close"], 20).ema_indicator()
    out["ema50"] = EMAIndicator(out["close"], 50).ema_indicator()
    
    # Momentum & Strength
    out["adx14"] = ADXIndicator(out["high"], out["low"], out["close"], 14).adx()
    out["rsi2"] = RSIIndicator(out["close"], 2).rsi()
    out["rsi14"] = RSIIndicator(out["close"], 14).rsi()
    out["atr14"] = AverageTrueRange(out["high"], out["low"], out["close"], 14).average_true_range()
    
    # Volume Baselines
    out["vol_sma20"] = out["volume"].rolling(20).mean()
    out["vol_sma50"] = out["volume"].rolling(50).mean()
    
    # Volatility / Range Compression (VCP detection)
    out["range"] = out["high"] - out["low"]
    out["range_sma5"] = out["range"].rolling(5).mean()
    out["range_sma20"] = out["range"].rolling(20).mean()
    
    # Structural Highs (Shifted by 1 so current bar is evaluated against prior resistance)
    out["high_20_prevbar"] = out["high"].rolling(20).max().shift(1)
    out["low_5_prevbar"] = out["low"].rolling(5).min().shift(1)
    out["high_252"] = out["close"].rolling(252).max()
    
    return out


def add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    """Returns 'trending' | 'range_bound' | 'high_vol_avoid'."""
    row = nifty_daily.iloc[-1]
    if india_vix_latest is not None and india_vix_latest > 22:
        return "high_vol_avoid"
    if row["close"] < row["sma200"]:
        return "high_vol_avoid"
    if row["close"] > row["sma50"] > row["sma200"] and row["adx14"] > 18:
        return "trending"
    return "range_bound"


def swing_setup_a_pullback(df: pd.DataFrame, price_floor: float = 50) -> bool:
    """
    REFINED MEAN-REVERSION PULLBACK:
    Requires primary uptrend + extreme oversold (RSI2 < 10) on drying volume,
    plus candle confirmation (close > open) to prevent buying falling knives.
    """
    r = df.iloc[-1]
    r_prev = df.iloc[-2]
    
    uptrend = r["close"] > r["sma200"] and r["close"] > r["sma50"] and r["adx14"] > 20
    oversold = r_prev["rsi2"] < 10 or r["rsi2"] < 15
    vol_dry_up = r["volume"] < r["vol_sma20"]
    candle_turnaround = r["close"] >= r["open"]  # Closed green or neutral
    
    return bool(uptrend and oversold and vol_dry_up and candle_turnaround and r["close"] > price_floor)


def swing_setup_b_watchlist(df: pd.DataFrame, price_floor: float = 50) -> bool:
    """
    VCP COILING PHASE:
    Strong momentum (EMA20 > EMA50, ADX > 25, RSI > 60) coiling tightly 
    within 5% of its 20-day high with contracting daily range (volatility squeeze).
    """
    r = df.iloc[-1]
    
    momentum = r["ema20"] > r["ema50"] and r["adx14"] > 25 and r["rsi14"] > 60
    near_high = (0.95 * r["high_20_prevbar"]) <= r["close"] <= r["high_20_prevbar"]
    range_compression = r["range_sma5"] < 0.85 * r["range_sma20"]  # Volatility squeeze
    
    return bool(momentum and near_high and range_compression and r["close"] > price_floor)


def swing_setup_b_trigger(df: pd.DataFrame, price_floor: float = 50) -> bool:
    """
    CONFIRMED MOMENTUM BREAKOUT:
    Closes definitively above 20-day high with 2x volume expansion and a top-quartile
    close ((Close - Low) / (High - Low) > 0.75) to invalidate bull traps.
    """
    r = df.iloc[-1]
    candle_range = (r["high"] - r["low"]) + 0.0001
    candle_strength = (r["close"] - r["low"]) / candle_range
    
    momentum = r["ema20"] > r["ema50"] and r["adx14"] > 25 and r["rsi14"] > 60
    breakout = r["close"] > r["high_20_prevbar"]
    volume_surge = r["volume"] > (2.0 * r["vol_sma20"])
    strong_close = candle_strength > 0.75
    
    return bool(momentum and breakout and volume_surge and strong_close and r["close"] > price_floor)


def generate_trade_plan(df: pd.DataFrame, setup_type: str) -> dict:
    """
    Computes exact entry price, initial stop-loss, and multi-tier targets.
    """
    r = df.iloc[-1]
    entry = float(r["close"])
    atr = float(r["atr14"]) if not np.isnan(r["atr14"]) else (entry * 0.02)
    
    if setup_type == "breakout":
        # Stop-Loss: Placed below breakout day's low or 1.5 ATR (whichever is tighter)
        candlestick_stop = float(r["low"]) - (0.1 * atr)
        atr_stop = entry - (1.5 * atr)
        stop_loss = max(candlestick_stop, atr_stop)
    else:  # pullback
        # Stop-Loss: Placed below the recent 5-day swing low
        stop_loss = min(float(r["low_5_prevbar"]), entry - (1.5 * atr))
        
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        risk_per_share = entry * 0.02
        stop_loss = entry - risk_per_share

    target_1 = round(entry + (1.5 * risk_per_share), 2)  # 1:1.5 R:R
    target_2 = round(entry + (2.5 * risk_per_share), 2)  # 1:2.5 R:R
    
    return {
        "entry_price": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target_1": target_1,
        "target_2": target_2,
        "risk_per_share": round(risk_per_share, 2),
        "risk_reward_t1": 1.5,
        "risk_reward_t2": 2.5
    }


def evaluate_swing(df: pd.DataFrame, regime: str, price_floor: float = 50) -> dict:
    df = add_daily_indicators(df)
    if len(df) < 252:
        return {"setup_a": False, "setup_b_watchlist": False, "setup_b_trigger": False,
                "note": "Need 252+ daily bars for reliable signals"}
                
    setup_a = swing_setup_a_pullback(df, price_floor)
    setup_b_watch = swing_setup_b_watchlist(df, price_floor)
    setup_b_trig = swing_setup_b_trigger(df, price_floor)
    
    trade_plan = None
    if setup_b_trig:
        trade_plan = generate_trade_plan(df, "breakout")
    elif setup_a:
        trade_plan = generate_trade_plan(df, "pullback")
        
    result = {
        "setup_a": setup_a,
        "setup_b_watchlist": setup_b_watch,
        "setup_b_trigger": setup_b_trig,
        "trade_plan": trade_plan
    }
    
    if regime == "high_vol_avoid":
        result["setup_a"] = False
        result["setup_b_watchlist"] = False
        result["setup_b_trigger"] = False
        result["trade_plan"] = None
        result["note"] = "Regime = high_vol_avoid (VIX > 22 or Nifty < 200 SMA): no swing entries allowed"
    elif regime == "range_bound":
        result["setup_b_watchlist"] = False
        result["setup_b_trigger"] = False
        if not setup_a:
            result["trade_plan"] = None
        result["note"] = "Regime = range_bound: Breakouts disabled. Setup A only (deploy half capital)"
    else:
        result["note"] = "Regime = trending: Both Breakouts & Pullbacks enabled (deploy full capital)"
        
    return result


def intraday_trend_mode(df: pd.DataFrame, price_floor: float = 50) -> dict:
    df = add_intraday_indicators(df)
    if len(df) < 20:
        return {"buy": False, "sell": False, "note": "Not enough bars yet today"}
    r = df.iloc[-1]
    buy = bool(r["close"] > r["ema20"] > r["ema50"] and r["adx14"] > 25 and 55 < r["rsi14"] < 72
               and r["close"] > r["vwap"] and r["volume"] > 2 * r["vol_sma20"] and r["close"] > price_floor)
    sell = bool(r["close"] < r["ema20"] < r["ema50"] and r["adx14"] > 25 and 28 < r["rsi14"] < 45
                and r["close"] < r["vwap"] and r["volume"] > 2 * r["vol_sma20"] and r["close"] > price_floor)
    return {"buy": buy, "sell": sell}


def intraday_range_mode(df: pd.DataFrame, price_floor: float = 50) -> dict:
    df = add_intraday_indicators(df)
    if len(df) < 20:
        return {"fade_short": False, "note": "Not enough bars yet today"}
    r = df.iloc[-1]
    fade = bool(r["close"] < r["vwap"] and (r["vwap"] - r["close"]) > 0.5 * r["atr14"]
                and r["adx14"] < 20 and r["rsi14"] < 35 and r["volume"] > 1.5 * r["vol_sma20"]
                and r["close"] > price_floor)
    return {"fade_short": fade}


def evaluate_intraday(df: pd.DataFrame, regime: str, price_floor: float = 50) -> dict:
    if regime == "high_vol_avoid":
        return {"note": "Regime = high_vol_avoid: no new intraday entries"}
    if regime == "trending":
        return intraday_trend_mode(df, price_floor)
    return intraday_range_mode(df, price_floor)


def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> int:
    risk_amount = capital * (risk_pct / 100)
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        return 0
    return int(risk_amount // per_share_risk)


def swing_signal_label(result: dict) -> str:
    if result.get("setup_a"):
        tp = result.get("trade_plan")
        return f"BUY (Pullback) | Entry: {tp['entry_price']} | SL: {tp['stop_loss']} | T1: {tp['target_1']}" if tp else "BUY (Setup A: Pullback)"
    if result.get("setup_b_trigger"):
        tp = result.get("trade_plan")
        return f"BUY (Breakout) | Entry: {tp['entry_price']} | SL: {tp['stop_loss']} | T1: {tp['target_1']}" if tp else "BUY (Setup B: Breakout)"
    if result.get("setup_b_watchlist"):
        return "WATCH (coiling/squeeze — near 20D high)"
    return "NO SIGNAL"


def intraday_signal_label(result: dict) -> str:
    if result.get("buy"):
        return "BUY"
    if result.get("sell"):
        return "SELL"
    if result.get("fade_short"):
        return "SELL (VWAP fade)"
    return "NO SIGNAL"
