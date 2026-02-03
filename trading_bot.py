"""
Binance Futures Trading Bot
Main orchestrator that ties together all components

Features:
- High win-rate AMR-VF strategy
- Multi-timeframe analysis
- Smart order execution with TWAP/VP algo orders
- Comprehensive risk management
- Telegram notifications
- Both live and paper trading modes
"""

import logging
import time
import threading
import queue
import signal as os_signal
import sys
import traceback
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from config import BotConfig, TradingMode, load_config_from_env
from binance_client import BinanceFuturesClient
from strategy import AMRVFStrategy, TradingSignal, SignalType
from risk_manager import RiskManager, Trade, TradeStatus
from order_executor import OrderExecutor, PaperTradeExecutor, ExecutionPlan


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send notifications to Telegram"""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)

        if self.enabled:
            import requests
            self.session = requests.Session()
            self.base_url = f"https://api.telegram.org/bot{token}"

    def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """Send text message to Telegram"""
        if not self.enabled:
            return False

        try:
            import requests
            response = self.session.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': text,
                    'parse_mode': parse_mode
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    def send_signal_alert(self, signal: TradingSignal):
        """Send formatted signal alert"""
        emoji = "🟢" if signal.signal_type == SignalType.LONG else "🔴"
        direction = "LONG" if signal.signal_type == SignalType.LONG else "SHORT"

        # Calculate risk/reward display
        risk_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
        reward_pct = abs(signal.take_profit_1 - signal.entry_price) / signal.entry_price * 100

        message = f"""
{emoji} <b>{direction} Signal - {signal.symbol}</b> {emoji}

<b>Strength:</b> {signal.strength}/10 | <b>Confidence:</b> {signal.confidence:.0%}
<b>Trend:</b> {signal.trend.value} | <b>Regime:</b> {signal.regime.value}

💰 <b>Entry:</b> ${signal.entry_price:.4f}
🛑 <b>Stop Loss:</b> ${signal.stop_loss:.4f} (-{risk_pct:.1f}%)
🎯 <b>TP1:</b> ${signal.take_profit_1:.4f} (+{reward_pct:.1f}%)
🎯 <b>TP2:</b> ${signal.take_profit_2:.4f}
🎯 <b>TP3:</b> ${signal.take_profit_3:.4f}

📊 <b>Indicators:</b>
• RSI: {signal.rsi:.1f}
• Stoch K/D: {signal.stoch_k:.1f}/{signal.stoch_d:.1f}
• Volume: {signal.volume_ratio:.1f}x
• R:R: 1:{signal.risk_reward_ratio:.1f}

📋 <b>Reasons:</b>
{chr(10).join(f"• {r}" for r in signal.reasons[:5])}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        self.send_message(message)

    def send_trade_update(self, trade: Trade, action: str, pnl: Optional[float] = None):
        """Send trade update notification"""
        emoji = "✅" if pnl and pnl > 0 else "❌" if pnl else "ℹ️"

        message = f"""
{emoji} <b>Trade {action}: {trade.symbol}</b>

<b>Side:</b> {trade.side}
<b>Entry:</b> ${trade.entry_price:.4f}
"""
        if pnl is not None:
            message += f"<b>PnL:</b> ${pnl:.2f}\n"

        if trade.exit_price:
            message += f"<b>Exit:</b> ${trade.exit_price:.4f}\n"

        self.send_message(message)

    def send_daily_summary(self, stats: Dict):
        """Send daily trading summary"""
        message = f"""
📊 <b>Daily Trading Summary</b>

<b>Trades:</b> {stats.get('total_trades', 0)}
<b>Win Rate:</b> {stats.get('win_rate', 0):.1f}%
<b>Total PnL:</b> ${stats.get('total_pnl', 0):.2f}

<b>Wins:</b> {stats.get('wins', 0)} | <b>Losses:</b> {stats.get('losses', 0)}
<b>Profit Factor:</b> {stats.get('profit_factor', 0):.2f}
<b>Avg Win:</b> ${stats.get('avg_win', 0):.2f}
<b>Avg Loss:</b> ${stats.get('avg_loss', 0):.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        self.send_message(message)


class TradingBot:
    """
    Main Trading Bot Orchestrator

    Coordinates:
    - Market scanning
    - Signal generation
    - Trade execution
    - Position management
    - Risk monitoring
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or load_config_from_env()

        # Initialize components
        self.client = BinanceFuturesClient(
            config=self.config.api,
            algo_config=self.config.algo
        )
        self.strategy = AMRVFStrategy(self.config.strategy)
        self.risk_manager = RiskManager(
            config=self.config.risk,
            db_path=self.config.db_file
        )

        # Initialize executor based on mode
        if self.config.mode == TradingMode.LIVE:
            self.executor = OrderExecutor(
                client=self.client,
                risk_manager=self.risk_manager,
                config=self.config
            )
            logger.warning("⚠️ LIVE TRADING MODE - Real orders will be placed!")
        else:
            self.executor = PaperTradeExecutor(
                client=self.client,
                risk_manager=self.risk_manager,
                config=self.config
            )
            logger.info("📝 Paper trading mode - No real orders")

        # Notifications
        self.notifier = TelegramNotifier(
            token=self.config.notifications.telegram_token,
            chat_id=self.config.notifications.telegram_chat_id
        )

        # State
        self.running = False
        self.scan_queue = queue.Queue()
        self.workers: List[threading.Thread] = []
        self.signals_cache: Dict[str, TradingSignal] = {}
        self.last_signal_time: Dict[str, datetime] = {}

        # Thread control
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        """Start the trading bot"""
        logger.info("🚀 Starting Trading Bot...")
        logger.info(f"Mode: {self.config.mode.value}")
        logger.info(f"Symbols: {len(self.config.symbols)}")

        self.running = True
        self._stop_event.clear()

        # Set leverage for all symbols
        self._initialize_symbols()

        # Get initial balance
        if self.config.mode == TradingMode.LIVE:
            balance = self.client.get_balance()
            if balance:
                self.risk_manager.update_balance(balance)
                logger.info(f"Account balance: ${balance:.2f}")
            else:
                logger.error("Failed to get account balance")
                return
        else:
            balance = self.executor.get_balance()
            self.risk_manager.update_balance(balance)
            logger.info(f"Paper balance: ${balance:.2f}")

        # Send startup notification
        self.notifier.send_message(
            f"🤖 <b>Trading Bot Started</b>\n\n"
            f"Mode: {self.config.mode.value}\n"
            f"Balance: ${balance:.2f}\n"
            f"Strategy: {self.config.strategy.name}\n"
            f"Symbols: {len(self.config.symbols)}"
        )

        # Start worker threads
        for i in range(3):  # 3 worker threads
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        # Start position monitor thread
        monitor = threading.Thread(
            target=self._position_monitor_loop,
            name="PositionMonitor",
            daemon=True
        )
        monitor.start()
        self.workers.append(monitor)

        # Main scan loop
        self._main_loop()

    def stop(self):
        """Stop the trading bot gracefully"""
        logger.info("Stopping Trading Bot...")
        self.running = False
        self._stop_event.set()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)

        # Save daily stats
        if self.config.mode == TradingMode.LIVE:
            balance = self.client.get_balance() or self.risk_manager.current_balance
        else:
            balance = self.executor.get_balance()

        self.risk_manager.update_daily_stats(balance)

        # Send summary
        stats = self.risk_manager.get_statistics(days=1)
        self.notifier.send_daily_summary(stats)

        logger.info("Trading Bot stopped")

    def _initialize_symbols(self):
        """Initialize leverage and other settings for symbols"""
        for symbol in self.config.symbols:
            try:
                success = self.client.set_leverage(symbol, self.config.default_leverage)
                if success:
                    logger.debug(f"Set leverage {self.config.default_leverage}x for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to set leverage for {symbol}: {e}")

    def _main_loop(self):
        """Main scanning loop"""
        last_full_scan = datetime.min

        while self.running and not self._stop_event.is_set():
            try:
                now = datetime.now()

                # Full scan every N seconds
                if (now - last_full_scan).total_seconds() >= self.config.full_scan_interval_seconds:
                    logger.info("Starting full market scan...")

                    # Add all symbols to scan queue
                    for symbol in self.config.symbols:
                        self.scan_queue.put(symbol)

                    last_full_scan = now

                # Quick scan more frequently
                time.sleep(self.config.scan_interval_seconds)

                # Check for daily reset
                if now.hour == 0 and now.minute < 5:
                    self.risk_manager.reset_daily_counters()

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(10)

    def _worker_loop(self):
        """Worker thread for scanning symbols"""
        while self.running and not self._stop_event.is_set():
            try:
                # Get symbol from queue with timeout
                try:
                    symbol = self.scan_queue.get(timeout=1)
                except queue.Empty:
                    continue

                self._analyze_symbol(symbol)

            except Exception as e:
                logger.error(f"Worker error: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1)

    def _analyze_symbol(self, symbol: str):
        """Analyze a single symbol for trading opportunities"""
        try:
            # Fetch kline data for multiple timeframes
            klines_dict = {}

            # Primary timeframe
            primary_tf = self.config.strategy.primary_timeframe
            klines = self.client.get_klines(symbol, primary_tf, 200)
            if not klines:
                logger.warning(f"{symbol}: Failed to fetch klines")
                return

            klines_dict[primary_tf] = klines

            # Confirmation timeframes
            for tf in self.config.strategy.confirmation_timeframes:
                tf_klines = self.client.get_klines(symbol, tf, 100)
                if tf_klines:
                    klines_dict[tf] = tf_klines

            # Analyze with multi-timeframe
            signal = self.strategy.analyze_multi_timeframe(symbol, klines_dict)

            if signal:
                logger.info(f"📊 SIGNAL FOUND: {symbol} {signal.signal_type.value} | Strength: {signal.strength}/10")
                self._process_signal(signal)
            else:
                # Log current price for visibility
                current_price = klines[-1]['close']
                logger.info(f"📈 {symbol}: ${current_price:.2f} - No signal (waiting for setup)")

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    def _process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        # Check signal cooldown (avoid duplicate signals)
        last_time = self.last_signal_time.get(signal.symbol)
        if last_time and (datetime.now() - last_time).total_seconds() < 3600:
            logger.debug(f"Signal cooldown for {signal.symbol}")
            return

        # Check if we can open trade
        can_trade, reason = self.risk_manager.can_open_trade(signal)
        if not can_trade:
            logger.info(f"Cannot open trade for {signal.symbol}: {reason}")
            return

        # Get current balance
        if isinstance(self.executor, PaperTradeExecutor):
            balance = self.executor.get_balance()
        else:
            balance = self.client.get_balance() or self.risk_manager.current_balance

        # Create execution plan
        plan = self.executor.create_execution_plan(signal, balance)
        if not plan:
            logger.warning(f"Failed to create execution plan for {signal.symbol}")
            return

        logger.info(
            f"Signal: {signal.symbol} {signal.signal_type.value} | "
            f"Strength: {signal.strength} | Value: ${plan.position_value:.2f}"
        )

        # Send notification
        if self.config.notifications.notify_signals:
            self.notifier.send_signal_alert(signal)

        # Execute trade
        with self._lock:
            success, trade, message = self.executor.execute_trade(plan)

        if success and trade:
            logger.info(f"Trade executed: {trade.trade_id}")
            self.last_signal_time[signal.symbol] = datetime.now()

            if self.config.notifications.notify_trades:
                self.notifier.send_trade_update(trade, "Opened")
        else:
            logger.error(f"Trade execution failed: {message}")

    def _position_monitor_loop(self):
        """Monitor open positions for trailing stops and exits"""
        while self.running and not self._stop_event.is_set():
            try:
                time.sleep(5)  # Check every 5 seconds

                # Get current prices
                prices = self.client.get_all_tickers()
                if not prices:
                    continue

                # Update paper positions if in paper mode
                if isinstance(self.executor, PaperTradeExecutor):
                    self.executor.update_positions(prices)
                    continue

                # Check active trades for trailing stops
                for trade_id, trade in list(self.risk_manager.active_trades.items()):
                    if trade.status != TradeStatus.OPEN:
                        continue

                    symbol = trade.symbol
                    if symbol not in prices:
                        continue

                    current_price = prices[symbol]

                    # Check trailing stop
                    new_stop = self.risk_manager.calculate_trailing_stop(
                        trade, current_price
                    )
                    if new_stop:
                        if isinstance(self.executor, OrderExecutor):
                            self.executor.update_trailing_stop(trade, new_stop)

                # Sync with exchange positions
                if isinstance(self.executor, OrderExecutor):
                    self.executor.sync_positions()

            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                time.sleep(5)

    def get_status(self) -> Dict:
        """Get current bot status"""
        if isinstance(self.executor, PaperTradeExecutor):
            balance = self.executor.get_balance()
            equity = self.executor.get_equity()
        else:
            balance = self.client.get_balance() or 0
            equity = balance + sum(
                t.unrealized_pnl
                for t in self.risk_manager.active_trades.values()
            )

        return {
            'running': self.running,
            'mode': self.config.mode.value,
            'balance': balance,
            'equity': equity,
            'active_trades': len(self.risk_manager.active_trades),
            'trades_today': self.risk_manager.trades_today,
            'daily_pnl': self.risk_manager.daily_pnl,
            'consecutive_losses': self.risk_manager.consecutive_losses
        }


def main():
    """Main entry point"""
    # Load configuration
    config = load_config_from_env()

    # Validate API credentials
    if config.mode == TradingMode.LIVE:
        if not config.api.api_key or not config.api.api_secret:
            logger.error("API credentials required for live trading")
            logger.error("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
            sys.exit(1)

    # Create and start bot
    bot = TradingBot(config)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal...")
        bot.stop()
        sys.exit(0)

    os_signal.signal(os_signal.SIGINT, signal_handler)
    os_signal.signal(os_signal.SIGTERM, signal_handler)

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        bot.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
