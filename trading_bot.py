"""
AI Crypto Trading Bot - Backtested Settings
- Threshold lowered to 30 (scalp) / 25 (swing) based on backtest
- Scans every 60 seconds so no signal is missed
- Gemini picks best coins daily
- 5% risk per trade, max 3 positions
"""

import asyncio
import json
import os
import re
import httpx
import pandas as pd
import numpy as np
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# ── YOUR KEYS ───────────────────────────────────────────────
ALPACA_API_KEY    = "PKEEPGET4QF47KCUWOKREWQEAI"
ALPACA_API_SECRET = "J39Y7S4EyzwGESEbBidAhqXx9ACRPhckCyKD4efqmnB4"
GEMINI_API_KEY    = "AIzaSyDiI_IOmusoBBt6jB44_GduFwRjYf_MTRQ"
# ────────────────────────────────────────────────────────────

# ── Backtested Settings ──────────────────────────────────────
RISK_PCT          = 0.05   # 5% of account per trade
MAX_POSITIONS     = 3      # max open trades at once
SCAN_INTERVAL     = 60     # scan every 60 seconds (never miss a signal)
SCALP_THRESHOLD   = 30     # proven in backtest: 54% win rate
SWING_THRESHOLD   = 25     # proven in backtest: +22% total PnL
SCALP_TP          = 0.015  # 1.5% take profit
SCALP_SL          = 0.008  # 0.8% stop loss
SWING_TP          = 0.04   # 4% take profit
SWING_SL          = 0.02   # 2% stop loss
# ────────────────────────────────────────────────────────────

trading     = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
data_client = CryptoHistoricalDataClient()

log_entries    = []
active_trades  = {}
selected_coins = []
last_ai_pick   = None
scan_count     = 0

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"time": timestamp, "level": level, "msg": msg}
    log_entries.append(entry)
    prefix = {"INFO": "✅", "WARN": "⚠️", "ERROR": "❌", "TRADE": "🔔", "AI": "🤖", "SCAN": "🔍"}.get(level, "•")
    print(f"[{timestamp}] {prefix} {msg}")
    with open("trading_log.json", "w") as f:
        json.dump(log_entries[-1000:], f, indent=2)

def get_bars(symbol, timeframe=TimeFrame.Minute, limit=100):
    try:
        req  = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, limit=limit)
        bars = data_client.get_crypto_bars(req)
        df   = bars.df
        if hasattr(df.index, 'levels'):
            df = df.xs(symbol, level=0)
        return df.reset_index()
    except Exception as e:
        log(f"Bar data error {symbol}: {e}", "ERROR")
        return None

def add_indicators(df):
    close = df['close']; high = df['high']; low = df['low']
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + (2 * std20)
    df['bb_lower'] = sma20 - (2 * std20)
    df['bb_mid']   = sma20
    # ATR / Volatility
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr']        = tr.rolling(14).mean()
    df['volatility'] = df['atr'] / close * 100
    # Volume
    df['vol_avg']   = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] / df['vol_avg']
    return df

def get_score_and_mode(df):
    """Returns (score, mode, reason_string)"""
    if df is None or len(df) < 30:
        return 0, 'swing', 'not enough data'
    df   = add_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score   = 0
    reasons = []
    mode    = 'scalp' if last['volatility'] > 1.2 else 'swing'

    # RSI
    rsi = last['rsi']
    if rsi < 35:
        score += 25; reasons.append(f"RSI oversold {rsi:.0f}")
    elif rsi < 45:
        score += 10; reasons.append(f"RSI low {rsi:.0f}")
    elif rsi > 70:
        score -= 25; reasons.append(f"RSI overbought {rsi:.0f}")
    elif rsi > 60:
        score -= 10; reasons.append(f"RSI high {rsi:.0f}")

    # MACD
    if prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']:
        score += 30; reasons.append("MACD bullish cross")
    elif prev['macd'] > prev['macd_signal'] and last['macd'] < last['macd_signal']:
        score -= 30; reasons.append("MACD bearish cross")
    elif last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist']:
        score += 10; reasons.append("MACD rising")
    elif last['macd_hist'] < 0 and last['macd_hist'] < prev['macd_hist']:
        score -= 10; reasons.append("MACD falling")

    # Bollinger Bands
    price = last['close']
    if price <= last['bb_lower']:
        score += 20; reasons.append("below BB lower")
    elif price >= last['bb_upper']:
        score -= 20; reasons.append("above BB upper")
    elif price > last['bb_mid'] and prev['close'] <= prev['bb_mid']:
        score += 15; reasons.append("crossed BB mid up")
    elif price < last['bb_mid'] and prev['close'] >= prev['bb_mid']:
        score -= 15; reasons.append("crossed BB mid down")

    # Volume spike confirms move
    if last['vol_spike'] > 1.5:
        score += 15; reasons.append(f"vol spike {last['vol_spike']:.1f}x")

    return score, mode, " | ".join(reasons) if reasons else "no signals"

def get_current_price(symbol):
    try:
        req   = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = data_client.get_crypto_latest_quote(req)
        return float(list(quote.values())[0].ask_price)
    except:
        return None

def ask_gemini_coins():
    global selected_coins, last_ai_pick
    today = datetime.now().date()
    if last_ai_pick == today and selected_coins:
        return selected_coins

    log("Asking Gemini to pick today's coins...", "AI")
    prompt = f"""You are a crypto trading analyst. Today is {datetime.now().strftime('%A %B %d %Y')}.
From: BTC, ETH, SOL, XRP, AVAX, DOGE, LINK, LTC, BCH, DOT
Pick TOP 4 with best short-term momentum and volatility today.
Reply ONLY with raw JSON array, no markdown:
["BTC","ETH","SOL","XRP"]"""

    try:
        r    = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15
        )
        text  = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        text  = re.sub(r"```json|```", "", text).strip()
        picks = [f"{p}/USD" for p in json.loads(text)][:4]
        selected_coins = picks
        last_ai_pick   = today
        log(f"Gemini picked: {[p.replace('/USD','') for p in picks]}", "AI")
        return picks
    except Exception as e:
        log(f"Gemini error: {e} — using BTC ETH SOL XRP", "ERROR")
        selected_coins = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
        return selected_coins

def place_buy(symbol, mode, score):
    try:
        cash  = float(trading.get_account().cash)
        price = get_current_price(symbol)
        if not price:
            log(f"Could not get price for {symbol}", "WARN")
            return

        qty      = round((cash * RISK_PCT) / price, 6)
        tp_pct   = SCALP_TP if mode == 'scalp' else SWING_TP
        sl_pct   = SCALP_SL if mode == 'scalp' else SWING_SL
        tp_price = round(price * (1 + tp_pct), 6)
        sl_price = round(price * (1 - sl_pct), 6)

        if qty <= 0:
            log(f"Not enough cash to trade {symbol}", "WARN")
            return

        result = trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        ))

        active_trades[symbol] = {
            "entry": price, "qty": qty,
            "tp": tp_price, "sl": sl_price,
            "mode": mode, "score": score,
            "opened_at": datetime.now().strftime("%H:%M:%S")
        }
        log(f"BUY {symbol} | ${price:.4f} | Qty:{qty} | TP:${tp_price:.4f} SL:${sl_price:.4f} | {mode.upper()} score={score}", "TRADE")

    except Exception as e:
        log(f"Order failed {symbol}: {e}", "ERROR")

def check_exits():
    for symbol, trade in list(active_trades.items()):
        price = get_current_price(symbol)
        if not price:
            continue
        pnl = (price - trade['entry']) / trade['entry'] * 100
        if price >= trade['tp']:
            try:
                trading.close_position(symbol)
                log(f"✅ TAKE PROFIT {symbol} | Entry:${trade['entry']:.4f} Exit:${price:.4f} | PnL:+{pnl:.2f}% | {trade['mode'].upper()}", "TRADE")
                del active_trades[symbol]
            except Exception as e:
                log(f"Close error {symbol}: {e}", "ERROR")
        elif price <= trade['sl']:
            try:
                trading.close_position(symbol)
                log(f"❌ STOP LOSS {symbol} | Entry:${trade['entry']:.4f} Exit:${price:.4f} | PnL:{pnl:.2f}% | {trade['mode'].upper()}", "TRADE")
                del active_trades[symbol]
            except Exception as e:
                log(f"Close error {symbol}: {e}", "ERROR")

async def run():
    global scan_count
    log("🤖 AI Trading Bot starting — backtested settings")
    log(f"Scalp threshold: {SCALP_THRESHOLD} | Swing threshold: {SWING_THRESHOLD}")
    log(f"Scanning every {SCAN_INTERVAL}s | Max {MAX_POSITIONS} positions | {RISK_PCT*100}% risk/trade")

    account = trading.get_account()
    log(f"Paper Account Cash: ${float(account.cash):,.2f}")

    while True:
        try:
            scan_count += 1
            coins = ask_gemini_coins()

            # Always check exits first
            if active_trades:
                check_exits()

            open_count = len(active_trades)

            for symbol in coins:
                if open_count >= MAX_POSITIONS:
                    break
                if symbol in active_trades:
                    continue

                df = get_bars(symbol, TimeFrame.Minute, limit=100)
                score, mode, reason = get_score_and_mode(df)
                threshold = SCALP_THRESHOLD if mode == 'scalp' else SWING_THRESHOLD

                # Log every 10 scans so terminal isn't too noisy
                if scan_count % 10 == 0:
                    log(f"{symbol.replace('/USD','')} score={score:+d} mode={mode} | {reason}", "SCAN")

                if score >= threshold:
                    log(f"🎯 SIGNAL {symbol} score={score} threshold={threshold} | {reason}")
                    place_buy(symbol, mode, score)
                    open_count += 1

            # Status every 10 scans (~10 mins)
            if scan_count % 10 == 0:
                cash = float(trading.get_account().cash)
                log(f"── Status | Open:{len(active_trades)} | Cash:${cash:,.2f} | Scan#{scan_count} ──")
                for sym, t in active_trades.items():
                    price = get_current_price(sym)
                    if price:
                        pnl = (price - t['entry']) / t['entry'] * 100
                        log(f"  {sym.replace('/USD','')} | Entry:${t['entry']:.4f} Now:${price:.4f} | PnL:{pnl:+.2f}% | {t['mode']}")

        except Exception as e:
            log(f"Loop error: {e}", "ERROR")

        await asyncio.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    asyncio.run(run())
