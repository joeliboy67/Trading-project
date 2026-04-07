"""
Automated Trading Bot — Bullish & Bearish Reversal Detection

Strategy:
  1. Identify support/resistance from recent swing lows/highs.
  2. When price bounces off previous support and breaks previous resistance
     → place a BUY with SL at support and TP at resistance above.
  3. Invalidate the setup if price breaks below support with bearish
     continuation (consecutive bearish closes below support).
  4. Additional confirmation from RSI divergence, MACD histogram reversal,
     candlestick patterns, and EMA trend context.

A trade signal fires when the support/resistance condition is met and at
least one confirming indicator agrees.
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
LOOKBACK_CANDLES = 250

# Support / resistance
SWING_WINDOW = 10          # bars each side to qualify as swing high/low
SR_TOUCH_TOLERANCE = 0.002 # 0.2 % proximity counts as a "touch"
BEARISH_CONT_BARS = 2      # consecutive bearish closes below support = invalid

# Signals
MIN_SIGNALS = 2  # S/R setup counts as 1; need at least 1 more confirming signal

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
    df["rsi"] = ta.momentum.rsi(df["close"], window=RSI_PERIOD)

    macd = ta.trend.MACD(
        df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["ema_short"] = ta.trend.ema_indicator(df["close"], window=EMA_SHORT)
    df["ema_long"] = ta.trend.ema_indicator(df["close"], window=EMA_LONG)

    return df


# ---------------------------------------------------------------------------
# Support & Resistance detection
# ---------------------------------------------------------------------------
def find_swing_levels(df: pd.DataFrame, window: int = SWING_WINDOW) -> tuple[list[float], list[float]]:
    """Return lists of swing-low (support) and swing-high (resistance) prices."""
    supports: list[float] = []
    resistances: list[float] = []

    for i in range(window, len(df) - window):
        # Swing low: lowest low in the surrounding window
        if df["low"].iloc[i] == df["low"].iloc[i - window : i + window + 1].min():
            supports.append(df["low"].iloc[i])
        # Swing high: highest high in the surrounding window
        if df["high"].iloc[i] == df["high"].iloc[i - window : i + window + 1].max():
            resistances.append(df["high"].iloc[i])

    return supports, resistances


def nearest_level_below(price: float, levels: list[float]) -> float | None:
    """Return the highest level that is still below the current price."""
    below = [lv for lv in levels if lv < price]
    return max(below) if below else None


def nearest_level_above(price: float, levels: list[float]) -> float | None:
    """Return the lowest level that is still above the current price."""
    above = [lv for lv in levels if lv > price]
    return min(above) if above else None


# ---------------------------------------------------------------------------
# Support / Resistance setup detection
# ---------------------------------------------------------------------------
def detect_sr_setup(df: pd.DataFrame) -> dict | None:
    """
    Detect the primary trade setup:
      - Price recently touched / bounced off a support level.
      - Price has now broken above the nearest resistance.
      - The setup is invalidated if price broke below support with bearish
        continuation (BEARISH_CONT_BARS consecutive bearish closes below support).

    Returns a dict with support, resistance, and direction, or None.
    """
    supports, resistances = find_swing_levels(df)
    if not supports or not resistances:
        log.info("  Not enough swing levels identified yet")
        return None

    curr_close = df["close"].iloc[-1]
    curr_low = df["low"].iloc[-1]

    # --- Find the previous support (nearest below current price) ---
    support = nearest_level_below(curr_close, supports)
    if support is None:
        log.info("  No support level below current price")
        return None

    # --- Find the previous resistance (nearest above current support) ---
    # We want the resistance the price needed to break through.
    # Look for the nearest resistance that is between support and current price,
    # meaning price has already broken above it.
    resistances_between = sorted(
        [r for r in resistances if support < r < curr_close]
    )

    # Also find the next resistance above current price for take-profit
    next_resistance = nearest_level_above(curr_close, resistances)

    if not resistances_between:
        log.info("  Price has not broken any resistance above support yet")
        return None

    broken_resistance = resistances_between[-1]  # most recent resistance that was broken

    # --- Check bounce: did price recently touch/approach the support? ---
    lookback = df.iloc[-20:]  # look at last 20 candles for the bounce
    tolerance = support * SR_TOUCH_TOLERANCE
    touched_support = (lookback["low"] <= support + tolerance).any()

    if not touched_support:
        log.info(f"  Price did not recently touch support at {support:.2f}")
        return None

    # --- Invalidation: bearish continuation below support ---
    recent = df.iloc[-BEARISH_CONT_BARS:]
    bearish_below_support = all(
        recent["close"].iloc[i] < support and recent["close"].iloc[i] < recent["open"].iloc[i]
        for i in range(len(recent))
    )
    if bearish_below_support:
        log.info(
            f"  INVALIDATED — {BEARISH_CONT_BARS} consecutive bearish closes "
            f"below support {support:.2f}"
        )
        return None

    # --- Setup confirmed ---
    # TP target: next resistance above current price, or if none, use a
    # measured-move projection (distance from support to broken resistance,
    # projected above broken resistance).
    if next_resistance:
        take_profit = next_resistance
    else:
        move = broken_resistance - support
        take_profit = broken_resistance + move

    log.info(f"  Support: {support:.2f}")
    log.info(f"  Broken resistance: {broken_resistance:.2f}")
    log.info(f"  Stop-loss: {support:.2f}  |  Take-profit: {take_profit:.2f}")

    return {
        "direction": "bullish",
        "support": support,
        "broken_resistance": broken_resistance,
        "stop_loss": support,
        "take_profit": take_profit,
    }


# ---------------------------------------------------------------------------
# Confirming reversal signals
# ---------------------------------------------------------------------------
def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    signals = {"bullish": 0, "bearish": 0}
    if len(df) < 2:
        return signals

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    body_curr = curr["close"] - curr["open"]
    body_prev = prev["close"] - prev["open"]

    if body_prev < 0 and body_curr > 0 and curr["close"] > prev["open"] and curr["open"] < prev["close"]:
        signals["bullish"] += 1
        log.info("  -> Bullish engulfing candle detected")

    if body_prev > 0 and body_curr < 0 and curr["close"] < prev["open"] and curr["open"] > prev["close"]:
        signals["bearish"] += 1
        log.info("  -> Bearish engulfing candle detected")

    candle_range = curr["high"] - curr["low"]
    if candle_range > 0:
        lower_shadow = min(curr["open"], curr["close"]) - curr["low"]
        upper_shadow = curr["high"] - max(curr["open"], curr["close"])
        body_size = abs(body_curr)
        if lower_shadow > 2 * body_size and upper_shadow < body_size and body_size < 0.3 * candle_range:
            signals["bullish"] += 1
            log.info("  -> Hammer candle detected")
        if upper_shadow > 2 * body_size and lower_shadow < body_size and body_size < 0.3 * candle_range:
            signals["bearish"] += 1
            log.info("  -> Shooting star candle detected")

    return signals


def detect_rsi_divergence(df: pd.DataFrame, window: int = 10) -> dict:
    signals = {"bullish": 0, "bearish": 0}
    if len(df) < window + 1:
        return signals

    recent = df.iloc[-window:]
    curr = df.iloc[-1]

    price_min_idx = recent["low"].idxmin()
    if curr["low"] <= recent["low"].min() and curr["rsi"] > recent.loc[price_min_idx, "rsi"]:
        signals["bullish"] += 1
        log.info("  -> Bullish RSI divergence detected")

    price_max_idx = recent["high"].idxmax()
    if curr["high"] >= recent["high"].max() and curr["rsi"] < recent.loc[price_max_idx, "rsi"]:
        signals["bearish"] += 1
        log.info("  -> Bearish RSI divergence detected")

    return signals


def detect_rsi_extremes(df: pd.DataFrame) -> dict:
    signals = {"bullish": 0, "bearish": 0}
    if len(df) < 3:
        return signals

    rsi_curr = df["rsi"].iloc[-1]
    rsi_prev = df["rsi"].iloc[-2]

    if rsi_prev < RSI_OVERSOLD and rsi_curr > rsi_prev:
        signals["bullish"] += 1
        log.info(f"  -> RSI rising from oversold ({rsi_prev:.1f} -> {rsi_curr:.1f})")

    if rsi_prev > RSI_OVERBOUGHT and rsi_curr < rsi_prev:
        signals["bearish"] += 1
        log.info(f"  -> RSI falling from overbought ({rsi_prev:.1f} -> {rsi_curr:.1f})")

    return signals


def detect_macd_reversal(df: pd.DataFrame) -> dict:
    signals = {"bullish": 0, "bearish": 0}
    if len(df) < 3:
        return signals

    hist_curr = df["macd_hist"].iloc[-1]
    hist_prev = df["macd_hist"].iloc[-2]
    hist_prev2 = df["macd_hist"].iloc[-3]

    if hist_prev2 > hist_prev and hist_curr > hist_prev and hist_curr < 0:
        signals["bullish"] += 1
        log.info("  -> MACD histogram bullish reversal")

    if hist_prev2 < hist_prev and hist_curr < hist_prev and hist_curr > 0:
        signals["bearish"] += 1
        log.info("  -> MACD histogram bearish reversal")

    return signals


def detect_ema_context(df: pd.DataFrame) -> dict:
    signals = {"bullish": 0, "bearish": 0}
    curr = df.iloc[-1]

    if pd.isna(curr["ema_long"]):
        return signals

    if curr["close"] > curr["ema_short"] and curr["ema_short"] < curr["ema_long"]:
        signals["bullish"] += 1
        log.info("  -> Price reclaiming short EMA in downtrend (bullish context)")

    if curr["close"] < curr["ema_short"] and curr["ema_short"] > curr["ema_long"]:
        signals["bearish"] += 1
        log.info("  -> Price losing short EMA in uptrend (bearish context)")

    return signals


# ---------------------------------------------------------------------------
# Signal aggregator
# ---------------------------------------------------------------------------
def aggregate_signals(df: pd.DataFrame) -> tuple[str, int, dict | None]:
    """
    Returns (direction, total_strength, sr_setup_or_None).

    The S/R bounce-and-break setup is the primary trigger. Confirming
    indicators add strength. A trade fires when total strength >= MIN_SIGNALS.
    """
    sr_setup = detect_sr_setup(df)

    confirming_detectors = [
        detect_candlestick_patterns,
        detect_rsi_divergence,
        detect_rsi_extremes,
        detect_macd_reversal,
        detect_ema_context,
    ]

    bullish_total = 0
    bearish_total = 0

    # S/R setup counts as 1 bullish signal
    if sr_setup and sr_setup["direction"] == "bullish":
        bullish_total += 1
        log.info("  S/R bounce-and-break setup ACTIVE (bullish +1)")

    for detector in confirming_detectors:
        result = detector(df)
        bullish_total += result["bullish"]
        bearish_total += result["bearish"]

    log.info(f"Signal tally — bullish: {bullish_total}, bearish: {bearish_total}")

    # Only go long if the S/R setup is present and we have enough confirmation
    if sr_setup and bullish_total >= MIN_SIGNALS and bullish_total > bearish_total:
        return "bullish", bullish_total, sr_setup

    # Allow standalone bearish signals (no S/R setup needed for shorts)
    if bearish_total >= MIN_SIGNALS and bearish_total > bullish_total:
        return "bearish", bearish_total, None

    return "neutral", 0, None


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------
def execute_trade(
    exchange: ccxt.Exchange,
    direction: str,
    strength: int,
    sr_setup: dict | None,
):
    """
    Place a market order with optional stop-loss and take-profit.

    When the S/R setup is present the SL/TP are derived from the detected
    support and resistance levels.
    """
    side = "buy" if direction == "bullish" else "sell"

    sl_price = sr_setup["stop_loss"] if sr_setup else None
    tp_price = sr_setup["take_profit"] if sr_setup else None

    log.info(
        f"{'🟢' if side == 'buy' else '🔴'} "
        f"{side.upper()} signal (strength {strength}) — "
        f"{SYMBOL} qty {TRADE_AMOUNT}"
    )
    if sl_price:
        log.info(f"  Stop-loss : {sl_price:.2f}")
    if tp_price:
        log.info(f"  Take-profit: {tp_price:.2f}")

    if not LIVE_TRADING:
        log.info("Paper-trade mode — order NOT sent to exchange")
        return None

    # --- Place primary market order ---
    order = exchange.create_market_order(SYMBOL, side, TRADE_AMOUNT)
    log.info(f"Order placed: {order['id']} — status: {order['status']}")

    # --- Place stop-loss order ---
    if sl_price:
        try:
            sl_order = exchange.create_order(
                symbol=SYMBOL,
                type="stop_market",
                side="sell",
                amount=TRADE_AMOUNT,
                params={"stopPrice": sl_price},
            )
            log.info(f"SL order placed: {sl_order['id']} @ {sl_price:.2f}")
        except ccxt.ExchangeError as e:
            log.warning(f"Could not place SL order: {e}")

    # --- Place take-profit order ---
    if tp_price:
        try:
            tp_order = exchange.create_order(
                symbol=SYMBOL,
                type="take_profit_market",
                side="sell",
                amount=TRADE_AMOUNT,
                params={"stopPrice": tp_price},
            )
            log.info(f"TP order placed: {tp_order['id']} @ {tp_price:.2f}")
        except ccxt.ExchangeError as e:
            log.warning(f"Could not place TP order: {e}")

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
    log.info(f"S/R swing window: {SWING_WINDOW} | Touch tolerance: {SR_TOUCH_TOLERANCE*100:.1f}%")
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

            direction, strength, sr_setup = aggregate_signals(df)

            if direction != "neutral":
                execute_trade(exchange, direction, strength, sr_setup)
            else:
                log.info("No trade signal — standing by")

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
