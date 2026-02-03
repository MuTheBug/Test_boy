"""
Advanced Trading Strategy Engine
Implements the AMR-VF (Adaptive Momentum Reversion with Volatility Filtering) Strategy

This is a high win-rate, low-drawdown strategy that combines:
1. Multi-timeframe trend alignment
2. Mean reversion entries within trending markets
3. Volatility regime filtering
4. Volume confirmation
5. Multiple confluence factors for high-probability setups
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config import StrategyConfig


logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class MarketRegime(Enum):
    """Market volatility regime"""
    LOW_VOLATILITY = "LOW_VOLATILITY"
    NORMAL = "NORMAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    EXTREME = "EXTREME"


class TrendDirection(Enum):
    """Trend direction"""
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    RANGING = "RANGING"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"


@dataclass
class IndicatorValues:
    """Container for all calculated indicator values"""
    # Price data
    closes: List[float] = field(default_factory=list)
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)
    opens: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)

    # Moving averages
    ema_fast: List[Optional[float]] = field(default_factory=list)
    ema_medium: List[Optional[float]] = field(default_factory=list)
    ema_slow: List[Optional[float]] = field(default_factory=list)
    sma_trend: List[Optional[float]] = field(default_factory=list)

    # Oscillators
    rsi: List[Optional[float]] = field(default_factory=list)
    stoch_k: List[Optional[float]] = field(default_factory=list)
    stoch_d: List[Optional[float]] = field(default_factory=list)

    # Volatility
    atr: List[Optional[float]] = field(default_factory=list)
    bb_upper: List[Optional[float]] = field(default_factory=list)
    bb_middle: List[Optional[float]] = field(default_factory=list)
    bb_lower: List[Optional[float]] = field(default_factory=list)

    # MACD
    macd_line: List[Optional[float]] = field(default_factory=list)
    macd_signal: List[Optional[float]] = field(default_factory=list)
    macd_histogram: List[Optional[float]] = field(default_factory=list)

    # Volume
    volume_sma: List[Optional[float]] = field(default_factory=list)
    volume_ratio: float = 1.0

    # Support/Resistance
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)


@dataclass
class TradingSignal:
    """Complete trading signal with all relevant information"""
    signal_type: SignalType
    symbol: str
    timeframe: str
    strength: int  # 1-10 scale
    confidence: float  # 0.0 - 1.0

    # Price levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float

    # Analysis
    trend: TrendDirection
    regime: MarketRegime
    reasons: List[str]
    confluence_factors: int

    # Indicators snapshot
    rsi: float
    stoch_k: float
    stoch_d: float
    atr: float
    volume_ratio: float

    # Risk metrics
    risk_reward_ratio: float
    estimated_win_rate: float

    # Timestamps
    timestamp: int


class TechnicalIndicators:
    """Calculate technical indicators without external dependencies"""

    @staticmethod
    def sma(data: List[float], period: int) -> List[Optional[float]]:
        """Simple Moving Average"""
        if len(data) < period:
            return [None] * len(data)

        result = [None] * (period - 1)
        for i in range(period - 1, len(data)):
            result.append(sum(data[i - period + 1:i + 1]) / period)
        return result

    @staticmethod
    def ema(data: List[float], period: int) -> List[Optional[float]]:
        """Exponential Moving Average"""
        if len(data) < period:
            return [None] * len(data)

        multiplier = 2 / (period + 1)
        result = [None] * (period - 1)

        # First EMA is SMA
        result.append(sum(data[:period]) / period)

        for i in range(period, len(data)):
            result.append((data[i] - result[-1]) * multiplier + result[-1])

        return result

    @staticmethod
    def rsi(data: List[float], period: int = 14) -> List[Optional[float]]:
        """Relative Strength Index"""
        if len(data) < period + 1:
            return [None] * len(data)

        gains = []
        losses = []

        for i in range(1, len(data)):
            diff = data[i] - data[i - 1]
            gains.append(max(diff, 0))
            losses.append(abs(min(diff, 0)))

        result = [None] * period

        # Initial average
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                result.append(100.0)
            else:
                rs = avg_gain / avg_loss
                result.append(100 - (100 / (1 + rs)))

        # Pad the beginning
        result = [None] + result
        return result

    @staticmethod
    def stochastic(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """Stochastic Oscillator"""
        k_values = []

        for i in range(len(closes)):
            if i < k_period - 1:
                k_values.append(None)
            else:
                highest = max(highs[i - k_period + 1:i + 1])
                lowest = min(lows[i - k_period + 1:i + 1])

                if highest == lowest:
                    k_values.append(50.0)
                else:
                    k = ((closes[i] - lowest) / (highest - lowest)) * 100
                    k_values.append(k)

        # Calculate %D (SMA of %K)
        k_valid = [k for k in k_values if k is not None]
        d_sma = TechnicalIndicators.sma(k_valid, d_period)

        d_values = []
        j = 0
        for k in k_values:
            if k is None:
                d_values.append(None)
            else:
                d_values.append(d_sma[j] if j < len(d_sma) else None)
                j += 1

        return k_values, d_values

    @staticmethod
    def atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> List[Optional[float]]:
        """Average True Range"""
        true_ranges = []

        for i in range(len(closes)):
            if i == 0:
                true_ranges.append(highs[i] - lows[i])
            else:
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1])
                )
                true_ranges.append(tr)

        return TechnicalIndicators.ema(true_ranges, period)

    @staticmethod
    def bollinger_bands(
        data: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        upper = []
        lower = []

        for i in range(len(data)):
            if sma[i] is None:
                upper.append(None)
                lower.append(None)
            else:
                std = np.std(data[max(0, i - period + 1):i + 1])
                upper.append(sma[i] + std_dev * std)
                lower.append(sma[i] - std_dev * std)

        return upper, sma, lower

    @staticmethod
    def macd(
        data: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)

        macd_line = []
        for i in range(len(data)):
            if ema_fast[i] is None or ema_slow[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])

        # Signal line
        macd_valid = [m for m in macd_line if m is not None]
        signal_ema = TechnicalIndicators.ema(macd_valid, signal)

        signal_line = []
        histogram = []
        j = 0
        for m in macd_line:
            if m is None:
                signal_line.append(None)
                histogram.append(None)
            else:
                sig = signal_ema[j] if j < len(signal_ema) else None
                signal_line.append(sig)
                histogram.append(m - sig if sig is not None else None)
                j += 1

        return macd_line, signal_line, histogram


class CandlePatterns:
    """Identify candlestick patterns"""

    @staticmethod
    def is_doji(open_p: float, high: float, low: float, close: float) -> bool:
        body = abs(close - open_p)
        range_hl = high - low
        return range_hl > 0 and body / range_hl < 0.1

    @staticmethod
    def is_hammer(open_p: float, high: float, low: float, close: float) -> bool:
        body = abs(close - open_p)
        range_hl = high - low
        if range_hl == 0:
            return False
        lower_shadow = min(open_p, close) - low
        upper_shadow = high - max(open_p, close)
        return lower_shadow > body * 2 and upper_shadow < body * 0.5

    @staticmethod
    def is_shooting_star(open_p: float, high: float, low: float, close: float) -> bool:
        body = abs(close - open_p)
        range_hl = high - low
        if range_hl == 0:
            return False
        lower_shadow = min(open_p, close) - low
        upper_shadow = high - max(open_p, close)
        return upper_shadow > body * 2 and lower_shadow < body * 0.5

    @staticmethod
    def is_engulfing_bullish(
        prev_open: float,
        prev_close: float,
        curr_open: float,
        curr_close: float
    ) -> bool:
        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open
        return prev_bearish and curr_bullish and curr_open <= prev_close and curr_close >= prev_open

    @staticmethod
    def is_engulfing_bearish(
        prev_open: float,
        prev_close: float,
        curr_open: float,
        curr_close: float
    ) -> bool:
        prev_bullish = prev_close > prev_open
        curr_bearish = curr_close < curr_open
        return prev_bullish and curr_bearish and curr_open >= prev_close and curr_close <= prev_open

    @staticmethod
    def is_morning_star(
        candles: List[Dict],
        threshold: float = 0.5
    ) -> bool:
        """Three-candle morning star pattern"""
        if len(candles) < 3:
            return False
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        body1 = c1['close'] - c1['open']
        body2 = abs(c2['close'] - c2['open'])
        body3 = c3['close'] - c3['open']
        range1 = c1['high'] - c1['low']

        return (
            body1 < 0 and  # First candle bearish
            body2 < abs(body1) * threshold and  # Small middle body
            body3 > 0 and  # Third candle bullish
            c3['close'] > c1['open'] + abs(body1) * 0.5  # Closes above midpoint of first
        )

    @staticmethod
    def is_evening_star(
        candles: List[Dict],
        threshold: float = 0.5
    ) -> bool:
        """Three-candle evening star pattern"""
        if len(candles) < 3:
            return False
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        body1 = c1['close'] - c1['open']
        body2 = abs(c2['close'] - c2['open'])
        body3 = c3['close'] - c3['open']

        return (
            body1 > 0 and  # First candle bullish
            body2 < abs(body1) * threshold and  # Small middle body
            body3 < 0 and  # Third candle bearish
            c3['close'] < c1['open'] - abs(body1) * 0.5  # Closes below midpoint
        )


class AMRVFStrategy:
    """
    Adaptive Momentum Reversion with Volatility Filtering (AMR-VF) Strategy

    Core Principles:
    1. Trade WITH the higher timeframe trend
    2. Enter on PULLBACKS/REVERSIONS within that trend
    3. Filter out unfavorable volatility regimes
    4. Require multiple confluence factors
    5. Use dynamic position sizing based on volatility

    Win Rate Target: 60-70%
    Max Drawdown Target: < 15%
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.indicators = TechnicalIndicators()
        self.patterns = CandlePatterns()

    def calculate_indicators(self, klines: List[Dict]) -> IndicatorValues:
        """Calculate all technical indicators from kline data"""
        if not klines or len(klines) < 100:
            return IndicatorValues()

        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        opens = [k['open'] for k in klines]
        volumes = [k['volume'] for k in klines]

        indicators = IndicatorValues(
            closes=closes,
            highs=highs,
            lows=lows,
            opens=opens,
            volumes=volumes
        )

        # Moving averages
        indicators.ema_fast = self.indicators.ema(closes, self.config.ema_fast)
        indicators.ema_medium = self.indicators.ema(closes, self.config.ema_medium)
        indicators.ema_slow = self.indicators.ema(closes, self.config.ema_slow)
        indicators.sma_trend = self.indicators.sma(closes, self.config.sma_trend)

        # Oscillators
        indicators.rsi = self.indicators.rsi(closes, self.config.rsi_period)
        indicators.stoch_k, indicators.stoch_d = self.indicators.stochastic(
            highs, lows, closes,
            self.config.stoch_k_period,
            self.config.stoch_d_period
        )

        # Volatility
        indicators.atr = self.indicators.atr(highs, lows, closes, self.config.atr_period)
        indicators.bb_upper, indicators.bb_middle, indicators.bb_lower = self.indicators.bollinger_bands(
            closes, self.config.bb_period, self.config.bb_std_dev
        )

        # MACD
        indicators.macd_line, indicators.macd_signal, indicators.macd_histogram = self.indicators.macd(
            closes, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )

        # Volume analysis
        indicators.volume_sma = self.indicators.sma(volumes, 20)
        if indicators.volume_sma[-1] and indicators.volume_sma[-1] > 0:
            indicators.volume_ratio = volumes[-1] / indicators.volume_sma[-1]

        # Find support/resistance levels
        indicators.support_levels, indicators.resistance_levels = self._find_sr_levels(
            highs, lows, closes
        )

        return indicators

    def _find_sr_levels(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels using swing points"""
        supports = []
        resistances = []
        window = self.config.sr_lookback

        for i in range(window, len(highs) - 5):
            # Swing high (resistance)
            if highs[i] == max(highs[i - window:i + window + 1]):
                resistances.append(highs[i])

            # Swing low (support)
            if lows[i] == min(lows[i - window:i + window + 1]):
                supports.append(lows[i])

        # Keep only recent and significant levels
        current_price = closes[-1]
        supports = sorted([s for s in supports if s < current_price], reverse=True)[:5]
        resistances = sorted([r for r in resistances if r > current_price])[:5]

        return supports, resistances

    def detect_trend(self, indicators: IndicatorValues) -> TrendDirection:
        """Detect current market trend using multiple indicators"""
        if not indicators.closes or len(indicators.closes) < 50:
            return TrendDirection.RANGING

        current_price = indicators.closes[-1]
        score = 0

        # EMA alignment
        ema_fast = indicators.ema_fast[-1]
        ema_medium = indicators.ema_medium[-1]
        ema_slow = indicators.ema_slow[-1]
        sma_trend = indicators.sma_trend[-1] if len(indicators.sma_trend) > 0 else None

        if all([ema_fast, ema_medium, ema_slow]):
            if ema_fast > ema_medium > ema_slow:
                score += 2
            elif ema_fast < ema_medium < ema_slow:
                score -= 2
            elif ema_fast > ema_medium:
                score += 1
            elif ema_fast < ema_medium:
                score -= 1

        # Price vs 200 SMA
        if sma_trend:
            if current_price > sma_trend * 1.02:
                score += 2
            elif current_price > sma_trend:
                score += 1
            elif current_price < sma_trend * 0.98:
                score -= 2
            elif current_price < sma_trend:
                score -= 1

        # MACD
        macd = indicators.macd_line[-1]
        signal = indicators.macd_signal[-1]
        if macd is not None and signal is not None:
            if macd > signal > 0:
                score += 1
            elif macd < signal < 0:
                score -= 1

        # Price momentum (20-period)
        if len(indicators.closes) >= 20:
            momentum = (indicators.closes[-1] - indicators.closes[-20]) / indicators.closes[-20]
            if momentum > 0.05:
                score += 1
            elif momentum < -0.05:
                score -= 1

        # Classify trend
        if score >= 4:
            return TrendDirection.STRONG_UPTREND
        elif score >= 2:
            return TrendDirection.UPTREND
        elif score <= -4:
            return TrendDirection.STRONG_DOWNTREND
        elif score <= -2:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.RANGING

    def detect_volatility_regime(self, indicators: IndicatorValues) -> MarketRegime:
        """Detect current volatility regime"""
        if not indicators.atr or len(indicators.atr) < 50:
            return MarketRegime.NORMAL

        current_atr = indicators.atr[-1]
        if current_atr is None:
            return MarketRegime.NORMAL

        # Calculate average ATR over last 50 periods
        atr_values = [a for a in indicators.atr[-50:] if a is not None]
        if not atr_values:
            return MarketRegime.NORMAL

        avg_atr = np.mean(atr_values)
        atr_ratio = current_atr / avg_atr

        if atr_ratio < self.config.low_volatility_threshold:
            return MarketRegime.LOW_VOLATILITY
        elif atr_ratio > self.config.high_volatility_threshold:
            return MarketRegime.EXTREME
        elif atr_ratio > self.config.optimal_volatility_max:
            return MarketRegime.HIGH_VOLATILITY
        else:
            return MarketRegime.NORMAL

    def calculate_confluence_score(
        self,
        indicators: IndicatorValues,
        signal_type: SignalType,
        trend: TrendDirection
    ) -> Tuple[int, List[str]]:
        """
        Calculate confluence score (0-10) based on multiple factors

        Returns (score, list of reasons)
        """
        score = 0
        reasons = []

        current_price = indicators.closes[-1]
        rsi = indicators.rsi[-1]
        stoch_k = indicators.stoch_k[-1]
        stoch_d = indicators.stoch_d[-1]
        macd_hist = indicators.macd_histogram[-1]
        volume_ratio = indicators.volume_ratio

        if signal_type == SignalType.LONG:
            # Factor 1: Trend alignment (2 points)
            if trend in [TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND]:
                score += 2
                reasons.append(f"Trend aligned ({trend.value})")

            # Factor 2: RSI oversold bounce (2 points)
            if rsi is not None:
                if rsi < self.config.rsi_extreme_oversold:
                    score += 2
                    reasons.append(f"RSI extreme oversold ({rsi:.1f})")
                elif rsi < self.config.rsi_oversold:
                    score += 1
                    reasons.append(f"RSI oversold ({rsi:.1f})")
                elif 40 <= rsi <= 60:
                    score += 1
                    reasons.append(f"RSI neutral zone ({rsi:.1f})")

            # Factor 3: Stochastic bullish cross (1 point)
            if stoch_k is not None and stoch_d is not None:
                if stoch_k > stoch_d and stoch_k < 30:
                    score += 1
                    reasons.append(f"Stochastic bullish cross in oversold")

            # Factor 4: MACD momentum (1 point)
            if macd_hist is not None:
                if len(indicators.macd_histogram) >= 2:
                    prev_hist = indicators.macd_histogram[-2]
                    if prev_hist is not None and macd_hist > prev_hist:
                        score += 1
                        reasons.append("MACD histogram increasing")

            # Factor 5: Price at support (1 point)
            if indicators.support_levels:
                nearest_support = indicators.support_levels[0]
                if abs(current_price - nearest_support) / current_price < 0.01:
                    score += 1
                    reasons.append(f"Price near support ({nearest_support:.4f})")

            # Factor 6: Bollinger Band bounce (1 point)
            bb_lower = indicators.bb_lower[-1]
            if bb_lower and current_price <= bb_lower * 1.005:
                score += 1
                reasons.append("Price at lower Bollinger Band")

            # Factor 7: Volume confirmation (1 point)
            if volume_ratio > self.config.volume_threshold_multiplier:
                score += 1
                reasons.append(f"Volume surge ({volume_ratio:.1f}x)")

            # Factor 8: Candlestick pattern (1 point)
            if len(indicators.closes) >= 2:
                if self.patterns.is_hammer(
                    indicators.opens[-1], indicators.highs[-1],
                    indicators.lows[-1], indicators.closes[-1]
                ):
                    score += 1
                    reasons.append("Hammer pattern detected")
                elif self.patterns.is_engulfing_bullish(
                    indicators.opens[-2], indicators.closes[-2],
                    indicators.opens[-1], indicators.closes[-1]
                ):
                    score += 1
                    reasons.append("Bullish engulfing pattern")

        elif signal_type == SignalType.SHORT:
            # Factor 1: Trend alignment (2 points)
            if trend in [TrendDirection.DOWNTREND, TrendDirection.STRONG_DOWNTREND]:
                score += 2
                reasons.append(f"Trend aligned ({trend.value})")

            # Factor 2: RSI overbought (2 points)
            if rsi is not None:
                if rsi > self.config.rsi_extreme_overbought:
                    score += 2
                    reasons.append(f"RSI extreme overbought ({rsi:.1f})")
                elif rsi > self.config.rsi_overbought:
                    score += 1
                    reasons.append(f"RSI overbought ({rsi:.1f})")

            # Factor 3: Stochastic bearish cross (1 point)
            if stoch_k is not None and stoch_d is not None:
                if stoch_k < stoch_d and stoch_k > 70:
                    score += 1
                    reasons.append("Stochastic bearish cross in overbought")

            # Factor 4: MACD momentum (1 point)
            if macd_hist is not None:
                if len(indicators.macd_histogram) >= 2:
                    prev_hist = indicators.macd_histogram[-2]
                    if prev_hist is not None and macd_hist < prev_hist:
                        score += 1
                        reasons.append("MACD histogram decreasing")

            # Factor 5: Price at resistance (1 point)
            if indicators.resistance_levels:
                nearest_resistance = indicators.resistance_levels[0]
                if abs(current_price - nearest_resistance) / current_price < 0.01:
                    score += 1
                    reasons.append(f"Price near resistance ({nearest_resistance:.4f})")

            # Factor 6: Bollinger Band rejection (1 point)
            bb_upper = indicators.bb_upper[-1]
            if bb_upper and current_price >= bb_upper * 0.995:
                score += 1
                reasons.append("Price at upper Bollinger Band")

            # Factor 7: Volume confirmation (1 point)
            if volume_ratio > self.config.volume_threshold_multiplier:
                score += 1
                reasons.append(f"Volume surge ({volume_ratio:.1f}x)")

            # Factor 8: Candlestick pattern (1 point)
            if len(indicators.closes) >= 2:
                if self.patterns.is_shooting_star(
                    indicators.opens[-1], indicators.highs[-1],
                    indicators.lows[-1], indicators.closes[-1]
                ):
                    score += 1
                    reasons.append("Shooting star pattern")
                elif self.patterns.is_engulfing_bearish(
                    indicators.opens[-2], indicators.closes[-2],
                    indicators.opens[-1], indicators.closes[-1]
                ):
                    score += 1
                    reasons.append("Bearish engulfing pattern")

        return min(score, 10), reasons

    def calculate_entry_exit_levels(
        self,
        signal_type: SignalType,
        current_price: float,
        atr: float,
        support_levels: List[float],
        resistance_levels: List[float]
    ) -> Dict[str, float]:
        """Calculate optimal entry, stop loss, and take profit levels"""
        levels = {}

        if signal_type == SignalType.LONG:
            # Entry at current price or slight pullback
            levels['entry'] = current_price

            # Stop loss below nearest support or ATR-based
            if support_levels:
                nearest_support = support_levels[0]
                atr_stop = current_price - (atr * self.config.atr_multiplier_sl)
                levels['stop_loss'] = min(nearest_support * 0.997, atr_stop)
            else:
                levels['stop_loss'] = current_price - (atr * self.config.atr_multiplier_sl)

            # Take profit levels
            levels['take_profit_1'] = current_price + (atr * self.config.atr_multiplier_tp1)
            levels['take_profit_2'] = current_price + (atr * self.config.atr_multiplier_tp2)
            levels['take_profit_3'] = current_price + (atr * self.config.atr_multiplier_tp3)

            # Adjust TP to resistance levels if nearby
            if resistance_levels:
                for i, tp_key in enumerate(['take_profit_1', 'take_profit_2', 'take_profit_3']):
                    for resistance in resistance_levels:
                        if levels[tp_key] * 0.99 <= resistance <= levels[tp_key] * 1.01:
                            levels[tp_key] = resistance * 0.998

        elif signal_type == SignalType.SHORT:
            levels['entry'] = current_price

            # Stop loss above nearest resistance or ATR-based
            if resistance_levels:
                nearest_resistance = resistance_levels[0]
                atr_stop = current_price + (atr * self.config.atr_multiplier_sl)
                levels['stop_loss'] = max(nearest_resistance * 1.003, atr_stop)
            else:
                levels['stop_loss'] = current_price + (atr * self.config.atr_multiplier_sl)

            # Take profit levels
            levels['take_profit_1'] = current_price - (atr * self.config.atr_multiplier_tp1)
            levels['take_profit_2'] = current_price - (atr * self.config.atr_multiplier_tp2)
            levels['take_profit_3'] = current_price - (atr * self.config.atr_multiplier_tp3)

            # Adjust TP to support levels if nearby
            if support_levels:
                for tp_key in ['take_profit_1', 'take_profit_2', 'take_profit_3']:
                    for support in support_levels:
                        if levels[tp_key] * 0.99 <= support <= levels[tp_key] * 1.01:
                            levels[tp_key] = support * 1.002

        return levels

    def analyze(
        self,
        symbol: str,
        klines: List[Dict],
        timeframe: str = "15m"
    ) -> Optional[TradingSignal]:
        """
        Main analysis function - generates trading signal if conditions are met

        Returns TradingSignal if a valid setup is found, None otherwise
        """
        if not klines or len(klines) < 100:
            logger.debug(f"{symbol}: Insufficient data for analysis")
            return None

        # Calculate indicators
        indicators = self.calculate_indicators(klines)

        # Detect market conditions
        trend = self.detect_trend(indicators)
        regime = self.detect_volatility_regime(indicators)

        # Filter out unfavorable regimes
        if regime == MarketRegime.EXTREME:
            logger.debug(f"{symbol}: Skipping - extreme volatility")
            return None

        if regime == MarketRegime.LOW_VOLATILITY:
            logger.debug(f"{symbol}: Skipping - low volatility (consolidation)")
            return None

        # Determine potential signal type based on trend and indicators
        signal_type = self._determine_signal_type(indicators, trend)

        if signal_type == SignalType.NEUTRAL:
            return None

        # Calculate confluence score
        confluence_score, reasons = self.calculate_confluence_score(
            indicators, signal_type, trend
        )

        # Require minimum confluence
        if confluence_score < self.config.min_signal_strength:
            logger.debug(
                f"{symbol}: Signal rejected - confluence {confluence_score} < {self.config.min_signal_strength}"
            )
            return None

        # Check volume confirmation if required
        if self.config.require_volume_confirmation:
            if indicators.volume_ratio < self.config.volume_threshold_multiplier * 0.8:
                logger.debug(f"{symbol}: Signal rejected - insufficient volume")
                return None

        # Get current values
        current_price = indicators.closes[-1]
        atr = indicators.atr[-1] if indicators.atr[-1] else current_price * 0.02
        rsi = indicators.rsi[-1] if indicators.rsi[-1] else 50
        stoch_k = indicators.stoch_k[-1] if indicators.stoch_k[-1] else 50
        stoch_d = indicators.stoch_d[-1] if indicators.stoch_d[-1] else 50

        # Calculate entry/exit levels
        levels = self.calculate_entry_exit_levels(
            signal_type,
            current_price,
            atr,
            indicators.support_levels,
            indicators.resistance_levels
        )

        # Calculate risk/reward
        risk = abs(levels['entry'] - levels['stop_loss'])
        reward = abs(levels['take_profit_1'] - levels['entry'])
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < self.config.min_risk_reward:
            logger.debug(f"{symbol}: Signal rejected - R:R {risk_reward:.2f} < {self.config.min_risk_reward}")
            return None

        # Estimate win rate based on confluence and trend alignment
        base_win_rate = 0.5
        win_rate_adjustment = (confluence_score - 5) * 0.03
        trend_bonus = 0.05 if trend in [TrendDirection.STRONG_UPTREND, TrendDirection.STRONG_DOWNTREND] else 0
        estimated_win_rate = min(0.75, max(0.4, base_win_rate + win_rate_adjustment + trend_bonus))

        # Calculate confidence
        confidence = confluence_score / 10.0

        logger.info(
            f"{symbol}: Signal generated - {signal_type.value} | "
            f"Confluence: {confluence_score}/10 | R:R: {risk_reward:.2f} | "
            f"Est. Win Rate: {estimated_win_rate:.0%}"
        )

        return TradingSignal(
            signal_type=signal_type,
            symbol=symbol,
            timeframe=timeframe,
            strength=confluence_score,
            confidence=confidence,
            entry_price=levels['entry'],
            stop_loss=levels['stop_loss'],
            take_profit_1=levels['take_profit_1'],
            take_profit_2=levels['take_profit_2'],
            take_profit_3=levels['take_profit_3'],
            trend=trend,
            regime=regime,
            reasons=reasons,
            confluence_factors=len(reasons),
            rsi=rsi,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            atr=atr,
            volume_ratio=indicators.volume_ratio,
            risk_reward_ratio=risk_reward,
            estimated_win_rate=estimated_win_rate,
            timestamp=klines[-1]['timestamp']
        )

    def _determine_signal_type(
        self,
        indicators: IndicatorValues,
        trend: TrendDirection
    ) -> SignalType:
        """Determine potential signal type based on conditions"""
        rsi = indicators.rsi[-1]
        stoch_k = indicators.stoch_k[-1]
        current_price = indicators.closes[-1]
        ema_fast = indicators.ema_fast[-1]

        if rsi is None or stoch_k is None or ema_fast is None:
            return SignalType.NEUTRAL

        # LONG conditions (mean reversion in uptrend)
        if trend in [TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND, TrendDirection.RANGING]:
            # Pullback to EMA or oversold conditions
            pullback_to_ema = current_price <= ema_fast * 1.005
            oversold = rsi < self.config.rsi_oversold or stoch_k < self.config.stoch_oversold

            if pullback_to_ema or oversold:
                return SignalType.LONG

        # SHORT conditions (mean reversion in downtrend)
        if trend in [TrendDirection.DOWNTREND, TrendDirection.STRONG_DOWNTREND, TrendDirection.RANGING]:
            # Rally to EMA or overbought conditions
            rally_to_ema = current_price >= ema_fast * 0.995
            overbought = rsi > self.config.rsi_overbought or stoch_k > self.config.stoch_overbought

            if rally_to_ema or overbought:
                return SignalType.SHORT

        return SignalType.NEUTRAL

    def analyze_multi_timeframe(
        self,
        symbol: str,
        klines_dict: Dict[str, List[Dict]]
    ) -> Optional[TradingSignal]:
        """
        Multi-timeframe analysis for higher probability trades

        Args:
            symbol: Trading symbol
            klines_dict: Dict of timeframe -> klines data
        """
        # Analyze primary timeframe
        primary_tf = self.config.primary_timeframe
        if primary_tf not in klines_dict:
            return None

        primary_signal = self.analyze(symbol, klines_dict[primary_tf], primary_tf)

        if not primary_signal:
            return None

        # Check confirmation timeframes
        confirmations = 0
        for tf in self.config.confirmation_timeframes:
            if tf in klines_dict:
                indicators = self.calculate_indicators(klines_dict[tf])
                htf_trend = self.detect_trend(indicators)

                # Confirm trend alignment
                if primary_signal.signal_type == SignalType.LONG:
                    if htf_trend in [TrendDirection.UPTREND, TrendDirection.STRONG_UPTREND]:
                        confirmations += 1
                elif primary_signal.signal_type == SignalType.SHORT:
                    if htf_trend in [TrendDirection.DOWNTREND, TrendDirection.STRONG_DOWNTREND]:
                        confirmations += 1

        # HTF confirmation is optional but boosts signal
        if confirmations == 0:
            # Still allow signal but reduce strength
            primary_signal.strength = max(1, primary_signal.strength - 1)
            primary_signal.reasons.append("No HTF confirmation (counter-trend)")
        else:
            # Boost signal strength with confirmations
            primary_signal.strength = min(10, primary_signal.strength + confirmations)
            primary_signal.confidence = min(1.0, primary_signal.confidence + 0.1 * confirmations)
            primary_signal.reasons.append(f"{confirmations} higher timeframe confirmation(s)")

        return primary_signal
