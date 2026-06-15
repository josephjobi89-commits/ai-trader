"""
AI Crypto Trading Bot - Fixed Alpaca Execution & Core Assets Only
Supported Tickers: BTC/USD, ETH/USD, SOL/USD, XRP/USD
Backtested settings: scalp threshold 30, swing threshold 25
Scans every 60s, 5% risk per trade, max 4 positions (one per coin)
"""

import asyncio
import json
import os
import pandas as pd
from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# ── KEYS FROM ENVIRONMENT VARIABLES ──────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
# ────────────────────────────────────────────────────────────

# Configured exclusively for your four requested core assets
COINS             = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD"]
RISK_PCT          = 0.05
MAX_POSITIONS     = 4   # Allows one position per coin across all four assets
SCAN_INTERVAL     = 60
SCALP_THRESHOLD   = 30
SWING_THRESHOLD   = 25
SCALP_TP          = 0.015
SCALP_SL          = 0.008
SWING_TP          = 0.04
SWING_SL          = 0.02

trading     = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)
data_client = CryptoHistoricalDataClient()

log_entries   = []
active_trades = {}
scan_count    = 0

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"time": timestamp, "level": level, "msg": msg}
    log_entries.append(entry)
    prefix = {"INFO": "✅", "WARN": "⚠️", "ERROR": "❌", "TRADE": "🔔", "SCAN": "🔍"}.get(level, "•")
    print(f"[{timestamp}] {prefix} {msg}")
    with open("trading_log.json", "w") as f:
        json.dump(log_entries[-1000:], f, indent=2)

def to_alpaca_symbol(symbol):
    """
    FIXED: Alpaca Trading API requires the slash format ('BTC/USD') 
    for crypto positions and order entries.
    """
    return symbol

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
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + (2 * std20)
    df['bb_lower'] = sma20 - (2 * std20)
    df['bb_mid']   = sma20
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr']        = tr.rolling(14).mean()
    df['volatility'] = df['atr'] / close * 100
    df['vol_avg']   = df['volume'].rolling(20).mean()
    df['vol_spike'] = df['volume'] / df['vol_avg']
    return df

def get_score_and_mode(df):
    if df is None or len(df) < 30:
        return 0, 'swing', 'not enough data'
    df   = add_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    score   = 0
    reasons = []
    mode    = 'scalp' if last['volatility'] > 1.2 else 'swing'

    rsi = last['rsi']
    if rsi < 35:
        score += 25; reasons.append(f"RSI oversold {rsi:.0f}")
    elif rsi < 45:
        score += 10; reasons.append(f"RSI low {rsi:.0f}")
    elif rsi > 70:
        score -= 25; reasons.append(f"RSI overbought {rsi:.0f}")
    elif rsi > 60:
        score -= 10; reasons.append(f"RSI high {rsi:.0f}")

    if prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']:
        score += 30; reasons.append("MACD bullish cross")
    elif prev['macd'] > prev['macd_signal'] and last['macd'] < last['macd_signal']:
        score -= 30; reasons.append("MACD bearish cross")
    elif last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist']:
        score += 10; reasons.append("MACD rising")
    elif last['macd_hist'] < 0 and last['macd_hist'] < prev['macd_hist']:
        score -= 10; reasons.append("MACD falling")

    price = last['close']
    if price <= last['bb_lower']:
        score += 20; reasons.append("below BB lower")
    elif price >= last['bb_upper']:
        score -= 20; reasons.append("above BB upper")
    elif price > last['bb_mid'] and prev['close'] <= prev['bb_mid']:
        score += 15; reasons.append("crossed BB mid up")
    elif price < last['bb_mid'] and prev['close'] >= prev['bb_mid']:
        score -= 15; reasons.append("crossed BB mid down")

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

        alpaca_symbol = to_alpaca_symbol(symbol)
        result = trading.submit_order(MarketOrderRequest(
            symbol=alpaca_symbol, qty=qty,
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
        alpaca_symbol = to_alpaca_symbol(symbol)

        if price >= trade['tp']:
            try:
                trading.close_position(alpaca_symbol)
                log(f"✅ TAKE PROFIT {symbol} | Entry:${trade['entry']:.4f} Exit:${price:.4f} | PnL:+{pnl:.2f}% | {trade['mode'].upper()}", "TRADE")
                del active_trades[symbol]
            except Exception as e:
                log(f"Close error {symbol} ({alpaca_symbol}): {e}", "ERROR")
        elif price <= trade['sl']:
            try:
                trading.close_position(alpaca_symbol)
                log(f"❌ STOP LOSS {symbol} | Entry:${trade['entry']:.4f} Exit:${price:.4f} | PnL:{pnl:.2f}% | {trade['mode'].upper()}", "TRADE")
                del active_trades[symbol]
            except Exception as e:
                log(f"Close error {symbol} ({alpaca_symbol}): {e}", "ERROR")

def sync_positions_from_alpaca():
    """On startup, rebuild active_trades from actual Alpaca positions"""
    try:
        positions = trading.get_all_positions()
        for p in positions:
            sym = p.symbol  # Could return as 'ETH/USD' or 'ETHUSD' depending on account configs
            if "/" not in sym and sym.endswith("USD"):
                sym = sym[:-3] + "/USD"

            if sym not in COINS:
                continue  # Skip outside tracking pairs

            if sym not in active_trades:
                entry = float(p.avg_entry_price)
                active_trades[sym] = {
                    "entry": entry,
                    "qty": float(p.qty),
                    "tp": round(entry * (1 + SWING_TP), 6),
                    "sl": round(entry * (1 - SWING_SL), 6),
                    "mode": "swing",
                    "score": 0,
                    "opened_at": "recovered"
                }
                log(f"Recovered existing position: {sym} | Entry:${entry:.4f}", "WARN")
    except Exception as e:
        log(f"Could not sync positions: {e}", "ERROR")

async def run():
    global scan_count
    log("🤖 Trading Bot initialized — BTC, ETH, SOL, & XRP Only")
    log(f"Scalp threshold: {SCALP_THRESHOLD} | Swing threshold: {SWING_THRESHOLD}")
    log(f"Scanning every {SCAN_INTERVAL}s | Max {MAX_POSITIONS} positions | {RISK_PCT*100}% risk/trade")

    account = trading.get_account()
    log(f"Paper Account Cash: ${float(account.cash):,.2f}")

    sync_positions_from_alpaca()

    while True:
        try:
            scan_count += 1

            if active_trades:
                check_exits()

            open_count = len(active_trades)

            for symbol in COINS:
                if open_count >= MAX_POSITIONS:
                    break
                if symbol in active_trades:
                    continue

                df = get_bars(symbol, TimeFrame.Minute, limit=100)
                score, mode, reason = get_score_and_mode(df)
                threshold = SCALP_THRESHOLD if mode == 'scalp' else SWING_THRESHOLD

                if scan_count % 10 == 0:
                    log(f"{symbol.replace('/USD','')} score={score:+d} mode={mode} | {reason}", "SCAN")

                if score >= threshold:
                    log(f"🎯 SIGNAL {symbol} score={score} threshold={threshold} | {reason}")
                    place_buy(symbol, mode, score)
                    open_count += 1

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
