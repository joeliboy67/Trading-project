"""
Automated Trading Bot — Bullish & Bearish Reversal Detection

Identifies reversals using a combination of:
  1. RSI divergence (price makes new high/low but RSI does not)
  2. MACD histogram reversal (histogram changes direction)
  3. Candlestick patterns (engulfing, hammer, shooting star)
  4. EMA trend context (50 & 200 EMA)

A trade signal fires when multiple indicators agree on a reversal.
"""

import os
import time
import logging
from datetime import datetime

import ccxt
import pandas as pd
import ta
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXCHANGE_ID = os.getenv("EXCHANGE", "binance")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "0.001"))
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

# Indicator settings
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 50
EMA_LONG = 200
LOOKBACK_CANDLES = 250  # candles to fetch

# How many confirming signals are needed to trigger a trade
MIN_SIGNALS = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("reversal-bot")


# ---------------------------------------------------------------------------
# Exchange helpers
# ---------------------------------------------------------------------------
def create_exchange() -> ccxt.Exchange:
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class(
        {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
        }
    )
    if not LIVE_TRADING:
        exchange.set_sandbox_mode(True)
        log.info("Running in SANDBOX / paper-trading mode")
    return exchange


def fetch_ohlcv(exchange: ccxt.Exchange) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LOOKBACK_CANDLES)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # RSI
    df["rsi"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)

    # MACD
    macd = ta.trend.MACD(
        df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # EMAs
    df["ema_short"] = ta.trend.ema_indicator(df["close"], window=EMA_SHORT)
    df["ema_long"] = ta.trend.ema_indicator(df["close"], window=EMA_LONG)

    return df


# ---------------------------------------------------------------------------
# Reversal detection signals
# ---------------------------------------------------------------------------
def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """Detect engulfing, hammer, and shooting-star patterns on the last candle."""
    signals = {"bullish": 0, "bearish": 0}
    if len(df) < 2:
        return signals

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    body_curr = curr["close"] - curr["open"]
    body_prev = prev["close"] - prev["open"]

    # --- Bullish engulfing ---
    if body_prev < 0 and body_curr > 0 and curr["close"] > prev["open"] and curr["open"] < prev["close"]:
        signals["bullish"] += 1
        log.info("  -> Bullish engulfing candle detected")

    # --- Bearish engulfing ---
    if body_prev > 0 and body_curr < 0 and curr["close"] < prev["open"] and curr["open"] > prev["close"]:
        signals["bearish"] += 1
        log.info("  -> Bearish engulfing candle detected")

    # --- Hammer (bullish) ---
    candle_range = curr["high"] - curr["low"]
    if candle_range > 0:
        lower_shadow = min(curr["open"], curr["close"]) - curr["low"]
        upper_shadow = curr["high"] - max(curr["open"], curr["close"])
        body_size = abs(body_curr)
        if lower_shadow > 2 * body_size and upper_shadow < body_size and body_size < 0.3 * candle_range:
            signals["bullish"] += 1
            log.info("  -> Hammer candle detected")

        # --- Shooting star (bearish) ---
        if upper_shadow > 2 * body_size and lower_shadow < body_size and body_size < 0.3 * candle_range:
            signals["bearish"] += 1
            log.info("  -> Shooting star candle detected")

    return signals


def detect_rsi_divergence(df: pd.DataFrame, window: int = 10) -> dict:
    """Simple RSI divergence: price makes new extreme but RSI does not."""
    signals = {"bullish": 0, "bearish": 0}
    if len(df) < window + 1:
        return signals

    recent = df.iloc[-window:]
    curr = df.iloc[-1]

    # Bullish divergence: price makes a lower low, RSI makes a higher low
    price_min_idx = recent["low"].idxmin()
    if curr["low"] <= recent["low"].min() and curr["rsi"] > recent.loc[price_min_idx, "rsi"]:
        signals["bullish"] += 1
        log.info("  -> Bullish RSI divergence detected")

    # Bearish divergence: price makes a higher high, RSI makes a lower high
    price_max_idx = recent["high"].idxmax()
    if curr["high"] >= recent["high"].max() and curr["rsi"] < recent.loc[price_max_idx, "rsi"]:
        signals["bearish"] += 1
        log.info("  -> Bearish RSI divergence detected")

    return signals


def detect_rsi_extremes(df: pd.DataFrame) -> dict:
    """RSI entering overbought/oversold and starting to turn."""
    signals = {"bullish": 0, "bearish": 0}
    if len(df) < 3:
        return signals

    rsi_curr = df["rsi"].iloc[-1]
    rsi_prev = df["rsi"].iloc[-2]

    # Oversold and rising → bullish reversal
    if rsi_prev < RSI_OVERSOLD and rsi_curr > rsi_prev:
        signals["bullish"] += 1
        log.info(f"  -> RSI rising from oversold ({rsi_prev:.1f} -> {rsi_curr:.1f})")

    # Overbought and falling → bearish reversal
    if rsi_prev > RSI_OVERBOUGHT and rsi_curr < rsi_prev:
        signals["bearish"] += 1
        log.info(f"  -> RSI falling from overbought ({rsi_prev:.1f} -> {rsi_curr:.1f})")

    return signals


def detect_macd_reversal(df: pd.DataFrame) -> dict:
    """MACD histogram flips direction."""
    signals = {"bullish": 0, "bearish": 0}
    if len(df) < 3:
        return signals

    hist_curr = df["macd_hist"].iloc[-1]
    hist_prev = df["macd_hist"].iloc[-2]
    hist_prev2 = df["macd_hist"].iloc[-3]

    # Histogram was falling and starts rising → bullish
    if hist_prev2 > hist_prev and hist_curr > hist_prev and hist_curr < 0:
        signals["bullish"] += 1
        log.info("  -> MACD histogram bullish reversal")

    # Histogram was rising and starts falling → bearish
    if hist_prev2 < hist_prev and hist_curr < hist_prev and hist_curr > 0:
        signals["bearish"] += 1
        log.info("  -> MACD histogram bearish reversal")

    return signals


def detect_ema_context(df: pd.DataFrame) -> dict:
    """Trend context from EMAs — supports reversals toward the trend."""
    signals = {"bullish": 0, "bearish": 0}
    curr = df.iloc[-1]

    if pd.isna(curr["ema_long"]):
        return signals

    # Price below both EMAs and starting to reclaim short EMA → bullish reversal
    if curr["close"] > curr["ema_short"] and curr["ema_short"] < curr["ema_long"]:
        signals["bullish"] += 1
        log.info("  -> Price reclaiming short EMA in downtrend (bullish context)")

    # Price above both EMAs and losing short EMA → bearish reversal
    if curr["close"] < curr["ema_short"] and curr["ema_short"] > curr["ema_long"]:
        signals["bearish"] += 1
        log.info("  -> Price losing short EMA in uptrend (bearish context)")

    return signals


# ---------------------------------------------------------------------------
# Signal aggregator
# ---------------------------------------------------------------------------
def aggregate_signals(df: pd.DataFrame) -> tuple[str, int]:
    """Run all detectors and return (direction, strength)."""
    detectors = [
        detect_candlestick_patterns,
        detect_rsi_divergence,
        detect_rsi_extremes,
        detect_macd_reversal,
        detect_ema_context,
    ]

    bullish_total = 0
    bearish_total = 0

    for detector in detectors:
        result = detector(df)
        bullish_total += result["bullish"]
        bearish_total += result["bearish"]

    log.info(f"Signal tally — bullish: {bullish_total}, bearish: {bearish_total}")

    if bullish_total >= MIN_SIGNALS and bullish_total > bearish_total:
        return "bullish", bullish_total
    if bearish_total >= MIN_SIGNALS and bearish_total > bullish_total:
        return "bearish", bearish_total

    return "neutral", 0


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------
def execute_trade(exchange: ccxt.Exchange, direction: str, strength: int):
    """Place a market order based on the detected reversal."""
    side = "buy" if direction == "bullish" else "sell"
    log.info(
        f"{'🟢' if side == 'buy' else '🔴'} "
        f"{side.upper()} signal (strength {strength}) — "
        f"{SYMBOL} qty {TRADE_AMOUNT}"
    )

    if not LIVE_TRADING:
        log.info("Paper-trade mode — order NOT sent to exchange")
        return None

    order = exchange.create_market_order(SYMBOL, side, TRADE_AMOUNT)
    log.info(f"Order placed: {order['id']} — status: {order['status']}")
    return order


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def run():
    exchange = create_exchange()
    sleep_seconds = TIMEFRAME_SECONDS.get(TIMEFRAME, 3600)

    log.info(f"Bot started — {SYMBOL} on {EXCHANGE_ID} ({TIMEFRAME})")
    log.info(f"Min signals required: {MIN_SIGNALS} | Live trading: {LIVE_TRADING}")
    log.info("-" * 60)

    while True:
        try:
            df = fetch_ohlcv(exchange)
            df = add_indicators(df)

            last = df.iloc[-1]
            log.info(
                f"Candle {last['timestamp']}  close={last['close']:.2f}  "
                f"RSI={last['rsi']:.1f}  MACD-H={last['macd_hist']:.4f}"
            )

            direction, strength = aggregate_signals(df)

            if direction != "neutral":
                execute_trade(exchange, direction, strength)
            else:
                log.info("No reversal signal — standing by")

            log.info(f"Sleeping {sleep_seconds}s until next candle...\n")
            time.sleep(sleep_seconds)

        except ccxt.NetworkError as e:
            log.warning(f"Network error: {e} — retrying in 30s")
            time.sleep(30)
        except ccxt.ExchangeError as e:
            log.error(f"Exchange error: {e}")
            time.sleep(60)
        except KeyboardInterrupt:
            log.info("Bot stopped by user")
            break


if __name__ == "__main__":
    run()
