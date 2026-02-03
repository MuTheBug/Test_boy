"""
HYPER SCALPER STRATEGY
Aggressive high-frequency strategy for rapid small account growth

Designed for:
- Small accounts ($5-100)
- High win rate (65%+)
- Quick profits, tight stops
- Multiple trades per day
- Compound growth focus
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
    Aggressive Scalping Strategy for Small Account Growth

    Core Principles:
    1. Trade WITH momentum (not against it)
    2. Quick entries on pullbacks in strong moves
    3. Tight stops (0.3-0.5%)
    4. Quick take profits (0.5-1%)
    5. High frequency = compound growth

    Indicators Used:
    - EMA 8/21 for trend
    - RSI 7 for momentum
    - Volume spike detection
    - Candle patterns (engulfing, pin bars)
    """

    def __init__(self):
        # EMA settings
        self.ema_fast = 8
        self.ema_slow = 21

        # RSI settings
        self.rsi_period = 7
        self.rsi_ob = 75  # Overbought
        self.rsi_os = 25  # Oversold

        # Risk settings (tight for scalping)
        self.stop_pct = 0.4  # 0.4% stop loss
        self.tp_pct = 0.8    # 0.8% take profit (2:1 RR)

        # Signal thresholds
        self.min_strength = 3
        self.volume_spike = 1.5  # 1.5x average volume

    def ema(self, data: List[float], period: int) -> List[float]:
        """Calculate EMA"""
        if len(data) < period:
            return [None] * len(data)

        mult = 2 / (period + 1)
        result = [None] * (period - 1)
        result.append(sum(data[:period]) / period)

        for i in range(period, len(data)):
            result.append((data[i] - result[-1]) * mult + result[-1])
        return result

    def rsi(self, data: List[float], period: int = 7) -> List[float]:
        """Calculate RSI"""
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

    def detect_candle_pattern(self, opens: List[float], highs: List[float],
                               lows: List[float], closes: List[float]) -> Tuple[str, int]:
        """
        Detect bullish/bearish candle patterns
        Returns (pattern_name, direction) where direction is 1 for bullish, -1 for bearish
        """
        if len(closes) < 3:
            return ("none", 0)

        # Current and previous candles
        o1, h1, l1, c1 = opens[-1], highs[-1], lows[-1], closes[-1]
        o2, h2, l2, c2 = opens[-2], highs[-2], lows[-2], closes[-2]

        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range1 = h1 - l1
        range2 = h2 - l2

        # Bullish Engulfing
        if c2 < o2 and c1 > o1 and o1 <= c2 and c1 >= o2 and body1 > body2:
            return ("bullish_engulfing", 1)

        # Bearish Engulfing
        if c2 > o2 and c1 < o1 and o1 >= c2 and c1 <= o2 and body1 > body2:
            return ("bearish_engulfing", -1)

        # Bullish Pin Bar (Hammer)
        if range1 > 0:
            lower_wick = min(o1, c1) - l1
            upper_wick = h1 - max(o1, c1)
            if lower_wick > body1 * 2 and upper_wick < body1 * 0.5:
                return ("hammer", 1)

        # Bearish Pin Bar (Shooting Star)
        if range1 > 0:
            lower_wick = min(o1, c1) - l1
            upper_wick = h1 - max(o1, c1)
            if upper_wick > body1 * 2 and lower_wick < body1 * 0.5:
                return ("shooting_star", -1)

        # Strong Momentum Candle (large body, small wicks)
        if range1 > 0 and body1 / range1 > 0.7:
            if c1 > o1:
                return ("momentum_bull", 1)
            else:
                return ("momentum_bear", -1)

        return ("none", 0)

    def analyze(self, klines: List[Dict]) -> Optional[ScalpTrade]:
        """
        Main analysis - generates scalp signal

        Entry Conditions for LONG:
        1. Price above EMA 21 (trend filter)
        2. EMA 8 > EMA 21 (momentum)
        3. RSI < 70 (not overbought)
        4. Pullback to EMA 8 OR bullish candle pattern
        5. Volume confirmation

        Entry Conditions for SHORT:
        1. Price below EMA 21 (trend filter)
        2. EMA 8 < EMA 21 (momentum)
        3. RSI > 30 (not oversold)
        4. Rally to EMA 8 OR bearish candle pattern
        5. Volume confirmation
        """
        if len(klines) < 50:
            return None

        # Extract data
        closes = [k['close'] for k in klines]
        opens = [k['open'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        volumes = [k['volume'] for k in klines]

        # Calculate indicators
        ema8 = self.ema(closes, self.ema_fast)
        ema21 = self.ema(closes, self.ema_slow)
        rsi = self.rsi(closes, self.rsi_period)

        # Current values
        price = closes[-1]
        curr_ema8 = ema8[-1]
        curr_ema21 = ema21[-1]
        curr_rsi = rsi[-1]
        prev_rsi = rsi[-2] if len(rsi) > 1 else None

        if None in [curr_ema8, curr_ema21, curr_rsi]:
            return None

        # Volume analysis
        avg_volume = sum(volumes[-20:]) / 20
        curr_volume = volumes[-1]
        volume_spike = curr_volume > avg_volume * self.volume_spike

        # Candle pattern
        pattern, direction = self.detect_candle_pattern(opens, highs, lows, closes)

        # Calculate strength score
        strength = 0
        reasons = []
        signal = ScalpSignal.NONE

        # ===== LONG SETUP =====
        if curr_ema8 > curr_ema21:  # Uptrend
            strength += 1
            reasons.append("EMA8 > EMA21 (uptrend)")

            # Price above EMA21
            if price > curr_ema21:
                strength += 1
                reasons.append("Price above EMA21")

            # RSI not overbought
            if curr_rsi < self.rsi_ob:
                strength += 1
                reasons.append(f"RSI {curr_rsi:.0f} not overbought")

            # RSI rising
            if prev_rsi and curr_rsi > prev_rsi:
                strength += 1
                reasons.append("RSI rising")

            # Pullback to EMA8 (within 0.3%)
            if abs(price - curr_ema8) / price < 0.003:
                strength += 2
                reasons.append("Pullback to EMA8")

            # Bullish candle pattern
            if direction == 1:
                strength += 2
                reasons.append(f"Pattern: {pattern}")

            # Volume spike
            if volume_spike:
                strength += 1
                reasons.append("Volume spike")

            # RSI oversold bounce
            if curr_rsi < 35 and prev_rsi and curr_rsi > prev_rsi:
                strength += 2
                reasons.append("RSI oversold bounce")

            if strength >= self.min_strength:
                signal = ScalpSignal.LONG

        # ===== SHORT SETUP =====
        elif curr_ema8 < curr_ema21:  # Downtrend
            strength += 1
            reasons.append("EMA8 < EMA21 (downtrend)")

            # Price below EMA21
            if price < curr_ema21:
                strength += 1
                reasons.append("Price below EMA21")

            # RSI not oversold
            if curr_rsi > self.rsi_os:
                strength += 1
                reasons.append(f"RSI {curr_rsi:.0f} not oversold")

            # RSI falling
            if prev_rsi and curr_rsi < prev_rsi:
                strength += 1
                reasons.append("RSI falling")

            # Rally to EMA8 (within 0.3%)
            if abs(price - curr_ema8) / price < 0.003:
                strength += 2
                reasons.append("Rally to EMA8")

            # Bearish candle pattern
            if direction == -1:
                strength += 2
                reasons.append(f"Pattern: {pattern}")

            # Volume spike
            if volume_spike:
                strength += 1
                reasons.append("Volume spike")

            # RSI overbought rejection
            if curr_rsi > 65 and prev_rsi and curr_rsi < prev_rsi:
                strength += 2
                reasons.append("RSI overbought rejection")

            if strength >= self.min_strength:
                signal = ScalpSignal.SHORT

        if signal == ScalpSignal.NONE:
            return None

        # Calculate entry, stop, take profit
        entry = price

        if signal == ScalpSignal.LONG:
            stop_loss = entry * (1 - self.stop_pct / 100)
            take_profit = entry * (1 + self.tp_pct / 100)
        else:
            stop_loss = entry * (1 + self.stop_pct / 100)
            take_profit = entry * (1 - self.tp_pct / 100)

        return ScalpTrade(
            signal=signal,
            symbol=klines[-1].get('symbol', 'UNKNOWN'),
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strength=strength,
            reason=" | ".join(reasons)
        )


class QuickBacktest:
    """Fast backtester for the scalping strategy"""

    def __init__(self, initial_balance: float = 5.0):
        self.initial_balance = initial_balance
        self.strategy = HyperScalper()

        # Trading params
        self.leverage = 50  # High leverage for small account
        self.risk_per_trade = 0.20  # 20% of account per trade (aggressive)
        self.fee_pct = 0.04  # 0.04% taker fee

    def run(self, klines: List[Dict]) -> Dict:
        """Run backtest and return results"""
        if len(klines) < 100:
            return {"error": "Insufficient data"}

        balance = self.initial_balance
        peak = balance
        max_dd = 0

        trades = []
        wins = 0
        losses = 0
        total_pnl = 0

        # Walk through data
        i = 50
        while i < len(klines) - 1:
            # Get historical slice for analysis
            history = klines[max(0, i-100):i+1]

            # Get signal
            trade = self.strategy.analyze(history)

            if trade and trade.signal != ScalpSignal.NONE:
                # Calculate position size
                position_value = balance * self.risk_per_trade * self.leverage
                qty = position_value / trade.entry

                # Simulate trade execution
                entry_price = trade.entry
                stop = trade.stop_loss
                tp = trade.take_profit

                # Check next candles for exit
                for j in range(i + 1, min(i + 20, len(klines))):  # Max 20 candles hold
                    candle = klines[j]
                    high = candle['high']
                    low = candle['low']

                    if trade.signal == ScalpSignal.LONG:
                        # Check stop loss
                        if low <= stop:
                            pnl = (stop - entry_price) * qty
                            pnl -= position_value * self.fee_pct / 100 * 2
                            balance += pnl
                            total_pnl += pnl
                            losses += 1
                            trades.append({
                                'side': 'LONG', 'entry': entry_price,
                                'exit': stop, 'pnl': pnl, 'result': 'STOP'
                            })
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
                            i = j
                            break
                    else:  # SHORT
                        # Check stop loss
                        if high >= stop:
                            pnl = (entry_price - stop) * qty
                            pnl -= position_value * self.fee_pct / 100 * 2
                            balance += pnl
                            total_pnl += pnl
                            losses += 1
                            trades.append({
                                'side': 'SHORT', 'entry': entry_price,
                                'exit': stop, 'pnl': pnl, 'result': 'STOP'
                            })
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
                            i = j
                            break
                else:
                    # Timeout - close at current price
                    close_price = klines[min(i + 20, len(klines) - 1)]['close']
                    if trade.signal == ScalpSignal.LONG:
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
                    i = min(i + 20, len(klines) - 1)

                # Track drawdown
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                if dd > max_dd:
                    max_dd = dd

                # Stop if blown
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
            'trades': trades[-20:]  # Last 20 trades
        }


def test_hyper_scalper(symbol: str = "BTCUSDT", balance: float = 5.0):
    """Quick test of the hyper scalper strategy"""
    from binance_client import BinanceFuturesClient
    from config import load_config_from_env

    config = load_config_from_env()
    client = BinanceFuturesClient(config.api)

    print(f"\n{'='*60}")
    print(f"  HYPER SCALPER BACKTEST - {symbol}")
    print(f"  Initial Balance: ${balance:.2f}")
    print(f"{'='*60}\n")

    # Fetch data - 5 minute candles for scalping
    print("Fetching 5m candles...")
    klines = client.get_klines(symbol, "5m", 1500)

    if not klines:
        print("Failed to fetch data")
        return

    print(f"Got {len(klines)} candles")

    # Run backtest
    bt = QuickBacktest(initial_balance=balance)
    results = bt.run(klines)

    # Print results
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
        emoji = "✅" if t['pnl'] > 0 else "❌"
        print(f"{emoji} {t['side']:5} | Entry: {t['entry']:.2f} | Exit: {t['exit']:.2f} | P&L: ${t['pnl']:+.2f} | {t['result']}")

    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    balance = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    test_hyper_scalper(symbol, balance)
