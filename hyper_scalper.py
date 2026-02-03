"""
HYPER SCALPER STRATEGY v5
TREND FOLLOWING PULLBACK + TRAILING STOP

CORE PRINCIPLE: NEVER fight the trend. Let winners run with trailing stop.

Key Rules:
1. Identify CLEAR trend using 200 EMA slope
2. Wait for pullback to key EMA (21 or 50)
3. Require STRONG reversal candle (engulfing/pin bar)
4. Wide stops (2x ATR) with 4:1 reward ratio
5. TRAILING STOP: Lock in profits as trade moves in your favor
6. Maximum 2 trades per day - QUALITY over quantity

Trailing Stop Logic:
- At 2x ATR profit: Move stop to breakeven
- At 3x ATR profit: Trail stop at 1.5x ATR behind price
- At 4x ATR profit: Trail stop at 1x ATR behind price (tight trail)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScalpSignal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class ScalpTrade:
    signal: ScalpSignal
    symbol: str
    entry: float
    stop_loss: float
    take_profit: float
    strength: int
    reason: str


class HyperScalper:
    """
    Trend Following Pullback Strategy v5

    Philosophy:
    - The trend is your friend - NEVER fight it
    - Pullbacks in trends offer low-risk entries
    - Strong reversal candles confirm the pullback is over
    - Wide stops prevent getting stopped out by noise
    - High reward ratio (4:1) means we only need 25% win rate to profit
    - TRAILING STOP lets winners run even further

    Entry Criteria (ALL must be met):
    1. Clear trend (200 EMA slope)
    2. Price pulled back to 21 or 50 EMA
    3. RSI in neutral zone (35-65) - not overbought/oversold
    4. Strong reversal candle pattern
    5. Volume above average
    """

    def __init__(self):
        # EMAs
        self.ema_fast = 21
        self.ema_slow = 50
        self.ema_trend = 200

        # RSI
        self.rsi_period = 14

        # ATR for stops
        self.atr_period = 14
        self.atr_stop_mult = 2.0   # Wide stop: 2x ATR
        self.atr_tp_mult = 8.0     # Target: 8x ATR (4:1 RR)

        # Trailing stop levels (in ATR multiples)
        self.trail_breakeven = 2.0   # Move to breakeven at 2x ATR profit
        self.trail_level1 = 3.0      # At 3x ATR profit, trail at 1.5x ATR
        self.trail_level2 = 4.0      # At 4x ATR profit, trail at 1x ATR (tight)

        # Minimum trend slope (% per 20 periods)
        self.min_trend_slope = 0.5

    def ema(self, data: List[float], period: int) -> List[float]:
        if len(data) < period:
            return [None] * len(data)
        mult = 2 / (period + 1)
        result = [None] * (period - 1)
        result.append(sum(data[:period]) / period)
        for i in range(period, len(data)):
            result.append((data[i] - result[-1]) * mult + result[-1])
        return result

    def sma(self, data: List[float], period: int) -> List[float]:
        if len(data) < period:
            return [None] * len(data)
        result = [None] * (period - 1)
        for i in range(period - 1, len(data)):
            result.append(sum(data[i - period + 1:i + 1]) / period)
        return result

    def rsi(self, data: List[float], period: int = 14) -> List[float]:
        if len(data) < period + 1:
            return [None] * len(data)
        gains, losses = [], []
        for i in range(1, len(data)):
            diff = data[i] - data[i-1]
            gains.append(max(diff, 0))
            losses.append(abs(min(diff, 0)))
        result = [None] * period
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                result.append(100)
            else:
                result.append(100 - (100 / (1 + avg_gain / avg_loss)))
        return [None] + result

    def atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        if len(closes) < period + 1:
            return [None] * len(closes)
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        result = [None] * period
        result.append(sum(trs[:period]) / period)
        for i in range(period, len(trs)):
            result.append((result[-1] * (period - 1) + trs[i]) / period)
        return [None] + result

    def get_trend_slope(self, ema_values: List[float], lookback: int = 20) -> float:
        """Calculate EMA slope as percentage change over lookback periods"""
        valid = [v for v in ema_values[-lookback:] if v is not None]
        if len(valid) < lookback:
            return 0
        return (valid[-1] - valid[0]) / valid[0] * 100

    def is_strong_bullish_candle(self, o, h, l, c) -> Tuple[bool, str]:
        """Check for strong bullish reversal patterns"""
        body = c - o
        range_hl = h - l
        if range_hl == 0:
            return False, ""

        body_pct = body / range_hl
        upper_wick = h - c
        lower_wick = o - l

        # Bullish engulfing-like (strong body, small wicks)
        if body > 0 and body_pct > 0.65:
            return True, "Strong bullish body"

        # Hammer (long lower wick, small upper wick)
        if body >= 0 and lower_wick > abs(body) * 2 and upper_wick < abs(body) * 0.5:
            return True, "Hammer"

        # Morning star setup (previous bearish, current bullish with gap-like behavior)
        if body > 0 and body_pct > 0.5 and lower_wick < body * 0.3:
            return True, "Bullish momentum"

        return False, ""

    def is_strong_bearish_candle(self, o, h, l, c) -> Tuple[bool, str]:
        """Check for strong bearish reversal patterns"""
        body = o - c
        range_hl = h - l
        if range_hl == 0:
            return False, ""

        body_pct = body / range_hl
        upper_wick = h - o
        lower_wick = c - l

        # Bearish engulfing-like (strong body, small wicks)
        if body > 0 and body_pct > 0.65:
            return True, "Strong bearish body"

        # Shooting star (long upper wick, small lower wick)
        if body >= 0 and upper_wick > abs(body) * 2 and lower_wick < abs(body) * 0.5:
            return True, "Shooting star"

        # Evening star setup
        if body > 0 and body_pct > 0.5 and upper_wick < body * 0.3:
            return True, "Bearish momentum"

        return False, ""

    def analyze(self, klines: List[Dict]) -> Optional[ScalpTrade]:
        """
        Trend Following Pullback Analysis

        Entry Rules:
        1. 200 EMA must have clear slope (trending)
        2. Price must have pulled back to 21 or 50 EMA
        3. RSI must be in neutral zone (not extreme)
        4. Current candle must be strong reversal
        5. Trade only in trend direction
        """
        if len(klines) < 220:
            return None

        closes = [k['close'] for k in klines]
        opens = [k['open'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        volumes = [k['volume'] for k in klines]

        # Calculate indicators
        ema21 = self.ema(closes, self.ema_fast)
        ema50 = self.ema(closes, self.ema_slow)
        ema200 = self.ema(closes, self.ema_trend)
        rsi = self.rsi(closes, self.rsi_period)
        atr = self.atr(highs, lows, closes, self.atr_period)

        # Current values
        price = closes[-1]
        e21, e50, e200 = ema21[-1], ema50[-1], ema200[-1]
        curr_rsi = rsi[-1]
        curr_atr = atr[-1]

        if None in [e21, e50, e200, curr_rsi, curr_atr]:
            return None

        # Current and previous candles
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]

        # Volume check
        avg_vol = sum(volumes[-20:]) / 20
        curr_vol = volumes[-1]
        good_volume = curr_vol >= avg_vol * 0.8  # At least 80% of average

        # Calculate trend slope
        trend_slope = self.get_trend_slope(ema200, 20)

        # ===== TREND DETECTION =====
        uptrend = trend_slope > self.min_trend_slope and price > e200
        downtrend = trend_slope < -self.min_trend_slope and price < e200

        if not uptrend and not downtrend:
            return None  # No clear trend - skip

        signal = ScalpSignal.NONE
        reasons = []

        # ===== UPTREND: Look for long entries =====
        if uptrend:
            # Check for pullback to EMA
            touched_ema21 = l <= e21 * 1.005 and l >= e21 * 0.99
            touched_ema50 = l <= e50 * 1.005 and l >= e50 * 0.99
            near_ema21 = abs(price - e21) / price < 0.008
            near_ema50 = abs(price - e50) / price < 0.012

            pullback_to_ema = touched_ema21 or touched_ema50 or near_ema21 or near_ema50

            if not pullback_to_ema:
                return None  # No pullback - skip

            # RSI should be neutral (pulled back but not oversold)
            rsi_ok = 35 <= curr_rsi <= 60
            if not rsi_ok:
                return None

            # Check for strong bullish candle
            is_bullish, pattern = self.is_strong_bullish_candle(o, h, l, c)
            if not is_bullish:
                return None

            # Check price closed above EMA (bounce confirmed)
            if c < e21 and c < e50:
                return None  # Didn't bounce

            # All conditions met!
            signal = ScalpSignal.LONG
            reasons = [
                f"Uptrend (slope: {trend_slope:.1f}%)",
                "Pullback to EMA",
                pattern,
                f"RSI: {curr_rsi:.0f}",
                "Volume OK" if good_volume else "Volume low"
            ]

        # ===== DOWNTREND: Look for short entries =====
        elif downtrend:
            # Check for rally to EMA
            touched_ema21 = h >= e21 * 0.995 and h <= e21 * 1.01
            touched_ema50 = h >= e50 * 0.995 and h <= e50 * 1.01
            near_ema21 = abs(price - e21) / price < 0.008
            near_ema50 = abs(price - e50) / price < 0.012

            rally_to_ema = touched_ema21 or touched_ema50 or near_ema21 or near_ema50

            if not rally_to_ema:
                return None  # No rally - skip

            # RSI should be neutral (rallied but not overbought)
            rsi_ok = 40 <= curr_rsi <= 65
            if not rsi_ok:
                return None

            # Check for strong bearish candle
            is_bearish, pattern = self.is_strong_bearish_candle(o, h, l, c)
            if not is_bearish:
                return None

            # Check price closed below EMA (rejection confirmed)
            if c > e21 and c > e50:
                return None  # Didn't reject

            # All conditions met!
            signal = ScalpSignal.SHORT
            reasons = [
                f"Downtrend (slope: {trend_slope:.1f}%)",
                "Rally to EMA",
                pattern,
                f"RSI: {curr_rsi:.0f}",
                "Volume OK" if good_volume else "Volume low"
            ]

        if signal == ScalpSignal.NONE:
            return None

        # Calculate stops with 3:1 RR
        if signal == ScalpSignal.LONG:
            stop_loss = price - (curr_atr * self.atr_stop_mult)
            take_profit = price + (curr_atr * self.atr_tp_mult)
        else:
            stop_loss = price + (curr_atr * self.atr_stop_mult)
            take_profit = price - (curr_atr * self.atr_tp_mult)

        return ScalpTrade(
            signal=signal,
            symbol=klines[-1].get('symbol', 'UNKNOWN'),
            entry=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strength=8,  # High quality setups only
            reason=" | ".join(reasons)
        )


class QuickBacktest:
    """Fast backtester with trailing stop support"""

    def __init__(self, initial_balance: float = 5.0):
        self.initial_balance = initial_balance
        self.strategy = HyperScalper()

        # Trading params - conservative for trend following
        self.leverage = 20  # Lower leverage with wider stops
        self.risk_per_trade = 0.10  # 10% of account per trade
        self.fee_pct = 0.04  # 0.04% taker fee

    def calculate_trailing_stop(self, entry: float, current_price: float,
                                 atr: float, is_long: bool, initial_stop: float) -> float:
        """
        Calculate trailing stop based on profit level

        Trailing Logic:
        - At 2x ATR profit: Move stop to breakeven
        - At 3x ATR profit: Trail at 1.5x ATR behind price
        - At 4x ATR profit: Trail at 1x ATR behind price (tight)
        """
        if is_long:
            profit_atr = (current_price - entry) / atr

            if profit_atr >= 4.0:
                # Tight trail at 1x ATR
                new_stop = current_price - (atr * 1.0)
            elif profit_atr >= 3.0:
                # Trail at 1.5x ATR
                new_stop = current_price - (atr * 1.5)
            elif profit_atr >= 2.0:
                # Move to breakeven
                new_stop = entry
            else:
                # Keep initial stop
                new_stop = initial_stop

            # Never move stop backwards
            return max(new_stop, initial_stop)
        else:
            # SHORT position
            profit_atr = (entry - current_price) / atr

            if profit_atr >= 4.0:
                new_stop = current_price + (atr * 1.0)
            elif profit_atr >= 3.0:
                new_stop = current_price + (atr * 1.5)
            elif profit_atr >= 2.0:
                new_stop = entry
            else:
                new_stop = initial_stop

            # Never move stop backwards (for short, lower is better)
            return min(new_stop, initial_stop)

    def run(self, klines: List[Dict]) -> Dict:
        """Run backtest with trailing stop"""
        if len(klines) < 250:
            return {"error": "Need at least 250 candles for 200 EMA"}

        balance = self.initial_balance
        peak = balance
        max_dd = 0

        trades = []
        wins = 0
        losses = 0
        total_pnl = 0

        # Track daily trades
        last_trade_candle = -100

        # Walk through data
        i = 220
        while i < len(klines) - 1:
            if i - last_trade_candle < 10:
                i += 1
                continue

            history = klines[max(0, i-250):i+1]
            trade = self.strategy.analyze(history)

            if trade and trade.signal != ScalpSignal.NONE:
                position_value = balance * self.risk_per_trade * self.leverage
                qty = position_value / trade.entry

                entry_price = trade.entry
                initial_stop = trade.stop_loss
                current_stop = initial_stop
                tp = trade.take_profit

                # Calculate ATR for trailing stop
                closes = [k['close'] for k in history]
                highs = [k['high'] for k in history]
                lows = [k['low'] for k in history]
                atr_values = self.strategy.atr(highs, lows, closes, 14)
                curr_atr = atr_values[-1] if atr_values[-1] else (trade.entry * 0.02)

                is_long = trade.signal == ScalpSignal.LONG
                max_price = entry_price  # Track best price for trailing
                exit_reason = ""

                # Check next candles for exit (max 80 candles for 4:1 RR)
                for j in range(i + 1, min(i + 80, len(klines))):
                    candle = klines[j]
                    high = candle['high']
                    low = candle['low']
                    close = candle['close']

                    if is_long:
                        # Update max price and trailing stop
                        if high > max_price:
                            max_price = high
                            current_stop = self.calculate_trailing_stop(
                                entry_price, max_price, curr_atr, True, initial_stop
                            )

                        # Check stop loss (including trailing)
                        if low <= current_stop:
                            exit_price = current_stop
                            pnl = (exit_price - entry_price) * qty
                            pnl -= position_value * self.fee_pct / 100 * 2
                            balance += pnl
                            total_pnl += pnl

                            if pnl > 0:
                                wins += 1
                                exit_reason = 'TRAIL_STOP'
                            else:
                                losses += 1
                                exit_reason = 'STOP'

                            trades.append({
                                'side': 'LONG', 'entry': entry_price,
                                'exit': exit_price, 'pnl': pnl, 'result': exit_reason
                            })
                            last_trade_candle = j
                            i = j
                            break

                        # Check take profit
                        if high >= tp:
                            pnl = (tp - entry_price) * qty
                            pnl -= position_value * self.fee_pct / 100 * 2
                            balance += pnl
                            total_pnl += pnl
                            wins += 1
                            trades.append({
                                'side': 'LONG', 'entry': entry_price,
                                'exit': tp, 'pnl': pnl, 'result': 'TP'
                            })
                            last_trade_candle = j
                            i = j
                            break

                    else:  # SHORT
                        # Update max price (min for short) and trailing stop
                        if low < max_price:
                            max_price = low
                            current_stop = self.calculate_trailing_stop(
                                entry_price, max_price, curr_atr, False, initial_stop
                            )

                        # Check stop loss (including trailing)
                        if high >= current_stop:
                            exit_price = current_stop
                            pnl = (entry_price - exit_price) * qty
                            pnl -= position_value * self.fee_pct / 100 * 2
                            balance += pnl
                            total_pnl += pnl

                            if pnl > 0:
                                wins += 1
                                exit_reason = 'TRAIL_STOP'
                            else:
                                losses += 1
                                exit_reason = 'STOP'

                            trades.append({
                                'side': 'SHORT', 'entry': entry_price,
                                'exit': exit_price, 'pnl': pnl, 'result': exit_reason
                            })
                            last_trade_candle = j
                            i = j
                            break

                        # Check take profit
                        if low <= tp:
                            pnl = (entry_price - tp) * qty
                            pnl -= position_value * self.fee_pct / 100 * 2
                            balance += pnl
                            total_pnl += pnl
                            wins += 1
                            trades.append({
                                'side': 'SHORT', 'entry': entry_price,
                                'exit': tp, 'pnl': pnl, 'result': 'TP'
                            })
                            last_trade_candle = j
                            i = j
                            break
                else:
                    # Timeout - close at current price
                    close_price = klines[min(i + 80, len(klines) - 1)]['close']
                    if is_long:
                        pnl = (close_price - entry_price) * qty
                    else:
                        pnl = (entry_price - close_price) * qty
                    pnl -= position_value * self.fee_pct / 100 * 2
                    balance += pnl
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    trades.append({
                        'side': trade.signal.value, 'entry': entry_price,
                        'exit': close_price, 'pnl': pnl, 'result': 'TIMEOUT'
                    })
                    last_trade_candle = min(i + 80, len(klines) - 1)
                    i = last_trade_candle

                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

                if balance <= 0:
                    break

            i += 1

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        return {
            'initial_balance': self.initial_balance,
            'final_balance': balance,
            'total_pnl': total_pnl,
            'total_pnl_pct': (balance - self.initial_balance) / self.initial_balance * 100,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'trades': trades[-20:]
        }


def test_hyper_scalper(symbol: str = "BTCUSDT", balance: float = 5.0, interval: str = "15m"):
    """Quick test of the hyper scalper strategy"""
    from binance_client import BinanceFuturesClient
    from config import load_config_from_env

    config = load_config_from_env()
    client = BinanceFuturesClient(config.api)

    print(f"\n{'='*60}")
    print(f"  TREND FOLLOWING v5 + TRAILING STOP - {symbol}")
    print(f"  Initial Balance: ${balance:.2f}")
    print(f"  Timeframe: {interval}")
    print(f"  Features: 4:1 RR + Trailing Stop")
    print(f"{'='*60}\n")

    print(f"Fetching {interval} candles...")
    klines = client.get_klines(symbol, interval, 1500)

    if not klines:
        print("Failed to fetch data")
        return

    print(f"Got {len(klines)} candles")

    bt = QuickBacktest(initial_balance=balance)
    results = bt.run(klines)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"Initial Balance:  ${results['initial_balance']:.2f}")
    print(f"Final Balance:    ${results['final_balance']:.2f}")
    print(f"Total P&L:        ${results['total_pnl']:.2f} ({results['total_pnl_pct']:+.1f}%)")
    print(f"\nTotal Trades:     {results['total_trades']}")
    print(f"Wins:             {results['wins']}")
    print(f"Losses:           {results['losses']}")
    print(f"Win Rate:         {results['win_rate']:.1f}%")
    print(f"Max Drawdown:     {results['max_drawdown']:.1f}%")

    print(f"\n--- Recent Trades ---")
    for t in results['trades'][-10:]:
        emoji = "+" if t['pnl'] > 0 else "-"
        print(f"{emoji} {t['side']:5} | Entry: {t['entry']:.2f} | Exit: {t['exit']:.2f} | P&L: ${t['pnl']:+.2f} | {t['result']}")

    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    balance = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    interval = sys.argv[3] if len(sys.argv) > 3 else "15m"
    test_hyper_scalper(symbol, balance, interval)
