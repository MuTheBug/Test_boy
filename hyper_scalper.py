"""
HYPER SCALPER STRATEGY v3
High Win Rate Mean Reversion + Momentum Combo

Key Principles:
1. Trade at extremes (Bollinger Band touches)
2. With trend confirmation (200 EMA)
3. RSI divergence for high-probability reversals
4. Multiple confluence = higher win rate
5. Strict entry criteria = quality over quantity
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
    High Win Rate Scalper v3

    Combines:
    1. Bollinger Band bounces (mean reversion at extremes)
    2. RSI divergence (strongest reversal signal)
    3. Trend alignment (200 EMA filter)
    4. Volume confirmation
    5. Candlestick patterns

    Only takes trades with 6+ confluence points
    """

    def __init__(self):
        # Trend EMAs
        self.ema_fast = 9
        self.ema_mid = 21
        self.ema_slow = 50
        self.ema_trend = 200  # Main trend filter

        # RSI
        self.rsi_period = 14

        # Bollinger Bands
        self.bb_period = 20
        self.bb_std = 2.0

        # Minimum score to trade (STRICT)
        self.min_strength = 6  # Need 6+ points for high win rate

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

    def bollinger_bands(self, data: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands - returns (upper, middle, lower)"""
        if len(data) < period:
            return [None] * len(data), [None] * len(data), [None] * len(data)

        upper = [None] * (period - 1)
        middle = [None] * (period - 1)
        lower = [None] * (period - 1)

        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            sma = sum(window) / period
            variance = sum((x - sma) ** 2 for x in window) / period
            std = variance ** 0.5

            middle.append(sma)
            upper.append(sma + std_dev * std)
            lower.append(sma - std_dev * std)

        return upper, middle, lower

    def detect_divergence(self, prices: List[float], rsi: List[float], lookback: int = 10) -> str:
        """
        Detect RSI divergence - one of the most reliable reversal signals

        Returns: 'bullish', 'bearish', or 'none'
        """
        if len(prices) < lookback or len(rsi) < lookback:
            return 'none'

        # Get valid RSI values
        valid_rsi = [r for r in rsi[-lookback:] if r is not None]
        if len(valid_rsi) < lookback:
            return 'none'

        prices_window = prices[-lookback:]
        rsi_window = valid_rsi

        # Find local lows and highs
        price_low_idx = prices_window.index(min(prices_window))
        price_high_idx = prices_window.index(max(prices_window))
        rsi_low_idx = rsi_window.index(min(rsi_window))
        rsi_high_idx = rsi_window.index(max(rsi_window))

        current_price = prices_window[-1]
        current_rsi = rsi_window[-1]

        # Bullish divergence: price making lower lows, RSI making higher lows
        if (price_low_idx > 2 and  # Recent low
            current_price < prices_window[0] and  # Price lower
            current_rsi > rsi_window[0]):  # RSI higher
            return 'bullish'

        # Bearish divergence: price making higher highs, RSI making lower highs
        if (price_high_idx > 2 and  # Recent high
            current_price > prices_window[0] and  # Price higher
            current_rsi < rsi_window[0]):  # RSI lower
            return 'bearish'

        return 'none'

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
        High Win Rate Analysis - Multiple Confluence Required

        Scoring System (need 6+ points):
        - Bollinger Band touch: +2 points
        - RSI divergence: +3 points (strongest signal)
        - RSI extreme (<30 or >70): +2 points
        - Trend alignment (200 EMA): +1 point
        - Bullish/Bearish candle pattern: +1 point
        - Hammer/Shooting star: +2 points
        - Volume confirmation: +1 point
        """
        if len(klines) < 220:  # Need enough data for 200 EMA
            return None

        closes = [k['close'] for k in klines]
        opens = [k['open'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        volumes = [k['volume'] for k in klines]

        # Calculate all indicators
        ema9 = self.ema(closes, self.ema_fast)
        ema21 = self.ema(closes, self.ema_mid)
        ema50 = self.ema(closes, self.ema_slow)
        ema200 = self.ema(closes, self.ema_trend)
        rsi = self.rsi(closes, self.rsi_period)
        atr = self.atr(highs, lows, closes, 14)
        bb_upper, bb_mid, bb_lower = self.bollinger_bands(closes, self.bb_period, self.bb_std)

        # Current values
        price = closes[-1]
        e9, e21, e50, e200 = ema9[-1], ema21[-1], ema50[-1], ema200[-1]
        curr_rsi = rsi[-1]
        prev_rsi = rsi[-2] if rsi[-2] else 50
        curr_atr = atr[-1]
        upper_bb = bb_upper[-1]
        lower_bb = bb_lower[-1]
        mid_bb = bb_mid[-1]

        if None in [e9, e21, e50, e200, curr_rsi, curr_atr, upper_bb, lower_bb]:
            return None

        # Current candle
        o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
        prev_o, prev_h, prev_l, prev_c = opens[-2], highs[-2], lows[-2], closes[-2]

        # Volume analysis
        avg_vol = sum(volumes[-20:]) / 20
        curr_vol = volumes[-1]
        high_volume = curr_vol > avg_vol * 1.2

        # Detect divergence
        divergence = self.detect_divergence(closes, rsi, 10)

        # ===== SCORING SYSTEM =====
        long_score = 0
        short_score = 0
        long_reasons = []
        short_reasons = []

        # 1. BOLLINGER BAND TOUCH (Mean Reversion at Extremes)
        bb_range = upper_bb - lower_bb
        near_lower = price <= lower_bb + bb_range * 0.05  # Within 5% of lower band
        near_upper = price >= upper_bb - bb_range * 0.05  # Within 5% of upper band
        touched_lower = l <= lower_bb
        touched_upper = h >= upper_bb

        if touched_lower or near_lower:
            long_score += 2
            long_reasons.append("At lower Bollinger Band")

        if touched_upper or near_upper:
            short_score += 2
            short_reasons.append("At upper Bollinger Band")

        # 2. RSI DIVERGENCE (Strongest Reversal Signal)
        if divergence == 'bullish':
            long_score += 3
            long_reasons.append("RSI bullish divergence")
        elif divergence == 'bearish':
            short_score += 3
            short_reasons.append("RSI bearish divergence")

        # 3. RSI EXTREME LEVELS
        if curr_rsi < 30:
            long_score += 2
            long_reasons.append(f"RSI oversold ({curr_rsi:.0f})")
        elif curr_rsi < 40:
            long_score += 1
            long_reasons.append(f"RSI low ({curr_rsi:.0f})")

        if curr_rsi > 70:
            short_score += 2
            short_reasons.append(f"RSI overbought ({curr_rsi:.0f})")
        elif curr_rsi > 60:
            short_score += 1
            short_reasons.append(f"RSI high ({curr_rsi:.0f})")

        # 4. TREND ALIGNMENT (Trade with 200 EMA)
        if price > e200:
            long_score += 1
            long_reasons.append("Above 200 EMA (uptrend)")
        else:
            short_score += 1
            short_reasons.append("Below 200 EMA (downtrend)")

        # 5. EMA ALIGNMENT (Short-term trend)
        if e9 > e21:
            long_score += 1
            long_reasons.append("Short EMAs bullish")
        elif e9 < e21:
            short_score += 1
            short_reasons.append("Short EMAs bearish")

        # 6. CANDLESTICK PATTERNS
        if self.is_bullish_candle(o, h, l, c):
            long_score += 1
            long_reasons.append("Bullish candle")
        if self.is_bearish_candle(o, h, l, c):
            short_score += 1
            short_reasons.append("Bearish candle")

        if self.is_hammer(prev_o, prev_h, prev_l, prev_c):
            long_score += 2
            long_reasons.append("Hammer pattern")
        if self.is_shooting_star(prev_o, prev_h, prev_l, prev_c):
            short_score += 2
            short_reasons.append("Shooting star")

        # Bullish engulfing
        if prev_c < prev_o and c > o and c > prev_o and o < prev_c:
            long_score += 2
            long_reasons.append("Bullish engulfing")
        # Bearish engulfing
        if prev_c > prev_o and c < o and c < prev_o and o > prev_c:
            short_score += 2
            short_reasons.append("Bearish engulfing")

        # 7. VOLUME CONFIRMATION
        if high_volume and c > o:
            long_score += 1
            long_reasons.append("High volume buying")
        if high_volume and c < o:
            short_score += 1
            short_reasons.append("High volume selling")

        # 8. RSI TURNING (Momentum shift)
        if curr_rsi > prev_rsi and curr_rsi < 60:
            long_score += 1
            long_reasons.append("RSI turning up")
        if curr_rsi < prev_rsi and curr_rsi > 40:
            short_score += 1
            short_reasons.append("RSI turning down")

        # ===== DETERMINE SIGNAL =====
        signal = ScalpSignal.NONE
        strength = 0
        reasons = []

        # Need minimum score AND no conflicting strong signals
        if long_score >= self.min_strength and long_score > short_score + 2:
            signal = ScalpSignal.LONG
            strength = long_score
            reasons = long_reasons
        elif short_score >= self.min_strength and short_score > long_score + 2:
            signal = ScalpSignal.SHORT
            strength = short_score
            reasons = short_reasons

        if signal == ScalpSignal.NONE:
            return None

        # Calculate stops based on ATR (tighter for higher win rate setups)
        # Use 1.2x ATR stop, 2.4x ATR target (2:1 RR)
        atr_stop = 1.2
        atr_tp = 2.4

        if signal == ScalpSignal.LONG:
            stop_loss = price - (curr_atr * atr_stop)
            take_profit = price + (curr_atr * atr_tp)
        else:
            stop_loss = price + (curr_atr * atr_stop)
            take_profit = price - (curr_atr * atr_tp)

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

        # Trading params - more conservative for higher win rate
        self.leverage = 30  # Moderate leverage
        self.risk_per_trade = 0.15  # 15% of account per trade
        self.fee_pct = 0.04  # 0.04% taker fee

    def run(self, klines: List[Dict]) -> Dict:
        """Run backtest and return results"""
        if len(klines) < 250:
            return {"error": "Need at least 250 candles for 200 EMA"}

        balance = self.initial_balance
        peak = balance
        max_dd = 0

        trades = []
        wins = 0
        losses = 0
        total_pnl = 0

        # Walk through data - start after 220 candles for 200 EMA
        i = 220
        while i < len(klines) - 1:
            # Get historical slice for analysis - need 220+ candles
            history = klines[max(0, i-250):i+1]

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


def test_hyper_scalper(symbol: str = "BTCUSDT", balance: float = 5.0, interval: str = "15m"):
    """Quick test of the hyper scalper strategy"""
    from binance_client import BinanceFuturesClient
    from config import load_config_from_env

    config = load_config_from_env()
    client = BinanceFuturesClient(config.api)

    print(f"\n{'='*60}")
    print(f"  HYPER SCALPER v3 BACKTEST - {symbol}")
    print(f"  Initial Balance: ${balance:.2f}")
    print(f"  Timeframe: {interval}")
    print(f"{'='*60}\n")

    # Fetch data
    print(f"Fetching {interval} candles...")
    klines = client.get_klines(symbol, interval, 1500)

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
    interval = sys.argv[3] if len(sys.argv) > 3 else "15m"
    test_hyper_scalper(symbol, balance, interval)
