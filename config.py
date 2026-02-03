"""
Configuration settings for the Binance Futures Trading Bot
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


class TradingMode(Enum):
    """Trading mode enum"""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class PositionMode(Enum):
    """Position mode for Binance Futures"""
    ONE_WAY = "one_way"  # BOTH position side
    HEDGE = "hedge"  # LONG/SHORT position sides


@dataclass
class APIConfig:
    """Binance API Configuration"""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False

    # Base URLs
    futures_base_url: str = "https://fapi.binance.com"
    futures_testnet_url: str = "https://testnet.binancefuture.com"

    # Algo order endpoints
    algo_twap_endpoint: str = "/sapi/v1/algo/futures/newOrderTwap"
    algo_vp_endpoint: str = "/sapi/v1/algo/futures/newOrderVp"
    algo_order_endpoint: str = "/fapi/v1/algoOrder"
    algo_query_endpoint: str = "/fapi/v1/algoOrder"
    algo_open_orders_endpoint: str = "/fapi/v1/openAlgoOrders"

    # Standard futures endpoints
    order_endpoint: str = "/fapi/v1/order"
    position_endpoint: str = "/fapi/v2/positionRisk"
    account_endpoint: str = "/fapi/v2/account"
    leverage_endpoint: str = "/fapi/v1/leverage"
    klines_endpoint: str = "/fapi/v1/klines"
    ticker_endpoint: str = "/fapi/v1/ticker/price"
    exchange_info_endpoint: str = "/fapi/v1/exchangeInfo"

    @property
    def base_url(self) -> str:
        return self.futures_testnet_url if self.testnet else self.futures_base_url


@dataclass
class RiskConfig:
    """Risk Management Configuration"""
    # Position sizing
    max_position_size_pct: float = 50.0  # Max 50% of account per trade (for small accounts)
    max_daily_loss_pct: float = 10.0  # Stop trading after 10% daily loss
    max_total_exposure_pct: float = 80.0  # Max 80% total exposure (for small accounts)
    max_trades_per_day: int = 10
    max_concurrent_positions: int = 3  # Fewer for small accounts

    # Risk per trade
    default_risk_pct: float = 5.0  # 5% risk per trade (higher for small accounts)
    max_risk_pct: float = 10.0  # Max 10% for small accounts
    min_risk_reward: float = 1.5  # Minimum 1:1.5 RR ratio

    # Minimum balance and position
    min_balance_usdt: float = 5.0  # Minimum balance to trade
    min_position_usdt: float = 5.0  # Minimum position size in USDT

    # Stop loss settings
    use_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 1.0  # Activate after 1% profit
    trailing_stop_callback_pct: float = 0.5  # Trail by 0.5%

    # Drawdown protection
    max_consecutive_losses: int = 5
    cooldown_after_max_losses_minutes: int = 60
    reduce_size_after_losses: bool = True
    size_reduction_factor: float = 0.5  # Reduce to 50% size after losses


@dataclass
class StrategyConfig:
    """Trading Strategy Configuration"""
    # Strategy name
    name: str = "AMR-VF"  # Adaptive Momentum Reversion with Volatility Filtering

    # Timeframes for multi-timeframe analysis
    primary_timeframe: str = "15m"
    confirmation_timeframes: List[str] = field(default_factory=lambda: ["1h", "4h"])

    # Entry conditions
    min_signal_strength: int = 5  # Minimum signal strength (out of 10) - higher for quality
    require_volume_confirmation: bool = False  # Disabled for more frequent signals
    volume_threshold_multiplier: float = 1.0  # Lowered threshold

    # RSI settings
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_extreme_oversold: float = 20.0
    rsi_extreme_overbought: float = 80.0

    # Stochastic settings
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    # Moving averages
    ema_fast: int = 9
    ema_medium: int = 21
    ema_slow: int = 50
    sma_trend: int = 200

    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # ATR for volatility
    atr_period: int = 14
    atr_multiplier_sl: float = 2.5  # Stop loss at 2.5x ATR (wider stop)
    atr_multiplier_tp1: float = 2.0  # TP1 at 2x ATR
    atr_multiplier_tp2: float = 4.0  # TP2 at 4x ATR
    atr_multiplier_tp3: float = 6.0  # TP3 at 6x ATR

    # Volatility regime
    low_volatility_threshold: float = 0.5  # Below 50% of average ATR
    high_volatility_threshold: float = 2.0  # Above 200% of average ATR
    optimal_volatility_min: float = 0.7
    optimal_volatility_max: float = 1.5

    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Support/Resistance
    sr_lookback: int = 50
    sr_min_touches: int = 2
    sr_tolerance_pct: float = 0.1

    # Take profit distribution
    tp1_size_pct: float = 40.0  # Close 40% at TP1
    tp2_size_pct: float = 30.0  # Close 30% at TP2
    tp3_size_pct: float = 30.0  # Close remaining 30% at TP3

    # Risk/Reward filter
    min_risk_reward: float = 1.0  # Minimum 1:1 R:R ratio (wider stops need lower RR)


@dataclass
class AlgoOrderConfig:
    """Algo Order Configuration"""
    # TWAP settings
    use_twap_for_large_orders: bool = True
    twap_threshold_usdt: float = 5000.0  # Use TWAP for orders > $5000
    twap_duration_seconds: int = 600  # 10 minutes default
    twap_min_duration: int = 300  # 5 minutes minimum
    twap_max_duration: int = 3600  # 1 hour maximum

    # VP (Volume Participation) settings
    use_vp_orders: bool = False
    vp_urgency: str = "LOW"  # LOW, MEDIUM, HIGH

    # General algo settings
    max_algo_orders: int = 10
    algo_order_timeout_seconds: int = 300


@dataclass
class NotificationConfig:
    """Notification Configuration"""
    telegram_enabled: bool = True
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # What to notify
    notify_signals: bool = True
    notify_trades: bool = True
    notify_stops: bool = True
    notify_errors: bool = True
    notify_daily_summary: bool = True


@dataclass
class BotConfig:
    """Main Bot Configuration"""
    # Trading mode
    mode: TradingMode = TradingMode.PAPER
    position_mode: PositionMode = PositionMode.ONE_WAY

    # Trading pairs (focus on majors - best backtest results)
    symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT"
    ])

    # Default leverage (higher for small accounts to meet min notional)
    default_leverage: int = 20
    max_leverage: int = 50

    # Scanning settings
    scan_interval_seconds: int = 60
    full_scan_interval_seconds: int = 300

    # Logging
    log_level: str = "INFO"
    log_file: str = "trading_bot.log"

    # Database
    db_file: str = "trading_data.db"

    # Sub-configs
    api: APIConfig = field(default_factory=APIConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    algo: AlgoOrderConfig = field(default_factory=AlgoOrderConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)


def load_config_from_env() -> BotConfig:
    """Load configuration from environment variables"""
    config = BotConfig()

    # API keys
    config.api.api_key = os.getenv("BINANCE_API_KEY", "")
    config.api.api_secret = os.getenv("BINANCE_API_SECRET", "")
    config.api.testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

    # Telegram
    config.notifications.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    config.notifications.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    # Trading mode
    mode_str = os.getenv("TRADING_MODE", "paper").lower()
    if mode_str == "live":
        config.mode = TradingMode.LIVE
    elif mode_str == "backtest":
        config.mode = TradingMode.BACKTEST
    else:
        config.mode = TradingMode.PAPER

    # Risk settings from env
    config.risk.max_position_size_pct = float(os.getenv("MAX_POSITION_SIZE_PCT", "2.0"))
    config.risk.max_daily_loss_pct = float(os.getenv("MAX_DAILY_LOSS_PCT", "5.0"))
    config.risk.default_risk_pct = float(os.getenv("DEFAULT_RISK_PCT", "1.0"))

    # Leverage
    config.default_leverage = int(os.getenv("DEFAULT_LEVERAGE", "10"))

    return config


# Default configuration instance
DEFAULT_CONFIG = BotConfig()
