"""
HYPER SCALPER STRATEGY v2
Conservative momentum strategy - Quality over Quantity

Key Changes:
- Only trade STRONG trends (not ranging)
- Wait for pullback + confirmation candle
- Higher win rate target (60%+)
- Fewer but better trades
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
    Conservative Momentum Scalper v2

    Only trades when:
    1. Strong trend (EMA alignment + momentum)
    2. Pullback occurred (price retraced to EMA)
    3. Confirmation candle (reversal pattern)
    4. Volume supports the move
    """

    def __init__(self):
        # Trend EMAs
        self.ema_fast = 9
        self.ema_mid = 21
        self.ema_slow = 55

        # RSI
        self.rsi_period = 14

        # Risk settings
        self.stop_pct = 0.6   # 0.6% stop loss
        self.tp_pct = 1.2     # 1.2% take profit (2:1 RR)

        # Minimum score to trade
        self.min_strength = 5  # Need 5+ points (stricter)

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
        """Average True Range"""
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

    def is_bullish_candle(self, o, h, l, c) -> bool:
        """Check if candle is bullish with good body"""
        body = c - o
        range_hl = h - l
        if range_hl == 0:
            return False
        return body > 0 and body / range_hl > 0.5

    def is_bearish_candle(self, o, h, l, c) -> bool:
        """Check if candle is bearish with good body"""
        body = o - c
        range_hl = h - l
        if range_hl == 0:
            return False
        return body > 0 and body / range_hl > 0.5

    def is_hammer(self, o, h, l, c) -> bool:
        """Bullish reversal pattern"""
        body = abs(c - o)
        range_hl = h - l
        if range_hl == 0 or body == 0:
            return False
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        return lower_wick > body * 2 and upper_wick < body * 0.5

    def is_shooting_star(self, o, h, l, c) -> bool:
        """Bearish reversal pattern"""
        body = abs(c - o)
        range_hl = h - l
        if range_hl == 0 or body == 0:
            return False
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        return upper_wick > body * 2 and lower_wick < body * 0.5

    def analyze(self, klines: List[Dict]) -> Optional[ScalpTrade]:
        """
        Conservative analysis - only strong setups
        """
        if len(klines) < 60:
            return None

        closes = [k['close'] for k in klines]
        opens = [k['open'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        volumes = [k['volume'] for k in klines]

        # Calculate indicators
        ema9 = self.ema(closes, self.ema_fast)
        ema21 = self.ema(closes, self.ema_mid)
        ema55 = self.ema(closes, self.ema_slow)
        rsi = self.rsi(closes, self.rsi_period)
        atr = self.atr(highs, lows, closes, 14)

        # Current values
        price = closes[-1]
        e9, e21, e55 = ema9[-1], ema21[-1], ema55[-1]
        curr_rsi = rsi[-1]
        prev_rsi = rsi[-2] if rsi[-2] else 50
        curr_atr = atr[-1]

        if None in [e9, e21, e55, curr_rsi, curr_atr]:
            return None

        # Current candle
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        prev_o, prev_h, prev_l, prev_c = opens[-2], highs[-2], lows[-2], closes[-2]

        # Volume analysis
        avg_vol = sum(volumes[-20:]) / 20
        curr_vol = volumes[-1]
        high_volume = curr_vol > avg_vol * 1.3

        # ===== DETECT TREND STRENGTH =====
        strength = 0
        reasons = []
        signal = ScalpSignal.NONE

        # STRONG UPTREND CHECK
        uptrend = e9 > e21 > e55
        downtrend = e9 < e21 < e55

        if uptrend:
            strength += 2
            reasons.append("Strong uptrend (EMA9>21>55)")

            # Price pulled back to EMA9 or EMA21
            near_ema9 = abs(price - e9) / price < 0.003
            near_ema21 = abs(price - e21) / price < 0.005
            touched_ema = l <= e9 * 1.002 or l <= e21 * 1.002

            if near_ema9 or touched_ema:
                strength += 2
                reasons.append("Pullback to EMA9")
            elif near_ema21:
                strength += 1
                reasons.append("Pullback to EMA21")

            # RSI not overbought and turning up
            if 40 < curr_rsi < 65:
                strength += 1
                reasons.append(f"RSI healthy ({curr_rsi:.0f})")
            if curr_rsi > prev_rsi and curr_rsi < 70:
                strength += 1
                reasons.append("RSI turning up")

            # Bullish confirmation candle
            if self.is_bullish_candle(o, h, l, c):
                strength += 1
                reasons.append("Bullish candle")
            if self.is_hammer(prev_o, prev_h, prev_l, prev_c):
                strength += 2
                reasons.append("Hammer pattern")

            # Volume confirmation
            if high_volume and c > o:
                strength += 1
                reasons.append("High volume buying")

            # Price above all EMAs
            if price > e9 and price > e21:
                strength += 1
                reasons.append("Price above EMAs")

            if strength >= self.min_strength:
                signal = ScalpSignal.LONG

        elif downtrend:
            strength += 2
            reasons.append("Strong downtrend (EMA9<21<55)")

            # Price rallied to EMA9 or EMA21
            near_ema9 = abs(price - e9) / price < 0.003
            near_ema21 = abs(price - e21) / price < 0.005
            touched_ema = h >= e9 * 0.998 or h >= e21 * 0.998

            if near_ema9 or touched_ema:
                strength += 2
                reasons.append("Rally to EMA9")
            elif near_ema21:
                strength += 1
                reasons.append("Rally to EMA21")

            # RSI not oversold and turning down
            if 35 < curr_rsi < 60:
                strength += 1
                reasons.append(f"RSI healthy ({curr_rsi:.0f})")
            if curr_rsi < prev_rsi and curr_rsi > 30:
                strength += 1
                reasons.append("RSI turning down")

            # Bearish confirmation candle
            if self.is_bearish_candle(o, h, l, c):
                strength += 1
                reasons.append("Bearish candle")
            if self.is_shooting_star(prev_o, prev_h, prev_l, prev_c):
                strength += 2
                reasons.append("Shooting star pattern")

            # Volume confirmation
            if high_volume and c < o:
                strength += 1
                reasons.append("High volume selling")

            # Price below all EMAs
            if price < e9 and price < e21:
                strength += 1
                reasons.append("Price below EMAs")

            if strength >= self.min_strength:
                signal = ScalpSignal.SHORT

        if signal == ScalpSignal.NONE:
            return None

        # Calculate stops based on ATR
        atr_mult = 1.5

        if signal == ScalpSignal.LONG:
            stop_loss = price - (curr_atr * atr_mult)
            take_profit = price + (curr_atr * atr_mult * 2)  # 2:1 RR
        else:
            stop_loss = price + (curr_atr * atr_mult)
            take_profit = price - (curr_atr * atr_mult * 2)

        return ScalpTrade(
            signal=signal,
            symbol=klines[-1].get('symbol', 'UNKNOWN'),
            entry=price,
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
