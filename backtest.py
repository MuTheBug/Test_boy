"""
Backtesting Module for the Trading Strategy
Test strategy performance on historical data
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from binance_client import BinanceFuturesClient
from strategy import AMRVFStrategy, TradingSignal, SignalType
from config import BotConfig, StrategyConfig, RiskConfig, load_config_from_env


logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A single backtest trade"""
    trade_id: int
    symbol: str
    side: str
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0

    # Partial exits tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    partial_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results"""
    symbol: str
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0

    win_rate: float = 0.0
    profit_factor: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0

    largest_win: float = 0.0
    largest_loss: float = 0.0

    avg_holding_time: str = ""

    sharpe_ratio: float = 0.0

    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """
    Strategy Backtester

    Tests the AMR-VF strategy on historical data
    """

    def __init__(
        self,
        config: Optional[BotConfig] = None,
        initial_balance: float = 10000.0,
        fee_pct: float = 0.04,  # 0.04% taker fee
        slippage_pct: float = 0.05  # 0.05% slippage
    ):
        self.config = config or load_config_from_env()
        self.strategy = AMRVFStrategy(self.config.strategy)
        self.risk_config = self.config.risk

        self.initial_balance = initial_balance
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

        # Client for fetching data (no API keys needed for public data)
        self.client = BinanceFuturesClient(self.config.api)

    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 1000
    ) -> List[Dict]:
        """Fetch historical kline data from Binance"""
        # Binance API max is 1500 candles per request
        limit = min(limit, 1500)
        logger.info(f"Fetching {limit} {interval} candles for {symbol}...")

        klines = self.client.get_klines(symbol, interval, limit)
        if not klines:
            logger.error(f"Failed to fetch data for {symbol}")
            return []

        logger.info(f"Fetched {len(klines)} candles")
        return klines

    def run_backtest(
        self,
        symbol: str,
        interval: str = "15m",
        lookback_candles: int = 1000
    ) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Timeframe (e.g., "15m", "1h")
            lookback_candles: Number of candles to test
        """
        # Fetch data
        klines = self.fetch_historical_data(symbol, interval, lookback_candles)
        if len(klines) < 200:
            logger.error("Insufficient data for backtesting")
            return BacktestResult(symbol=symbol, start_date="", end_date="",
                                  initial_balance=self.initial_balance,
                                  final_balance=self.initial_balance)

        # Initialize state
        balance = self.initial_balance
        equity_curve = [balance]
        peak_balance = balance
        max_drawdown = 0.0

        trades: List[BacktestTrade] = []
        active_trade: Optional[BacktestTrade] = None
        trade_id = 0

        consecutive_losses = 0

        # Walk through data
        min_candles = 200  # Need this many for indicators

        logger.info(f"Starting backtest: {symbol} {interval}")
        logger.info(f"Testing {len(klines) - min_candles} candles...")

        for i in range(min_candles, len(klines)):
            # Current candle
            candle = klines[i]
            current_price = candle['close']
            current_high = candle['high']
            current_low = candle['low']
            current_time = datetime.fromtimestamp(candle['timestamp'] / 1000)

            # Check active trade first
            if active_trade:
                trade_closed = self._check_trade_exit(
                    active_trade, current_high, current_low, current_price, current_time
                )

                if trade_closed:
                    # Apply fees
                    exit_value = active_trade.quantity * active_trade.exit_price
                    fees = exit_value * (self.fee_pct / 100) * 2  # Entry + exit
                    active_trade.pnl -= fees

                    # Update balance
                    balance += active_trade.pnl
                    equity_curve.append(balance)

                    # Track drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    drawdown = peak_balance - balance
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

                    # Track wins/losses
                    if active_trade.pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

                    trades.append(active_trade)
                    active_trade = None

            # Look for new signal if no active trade
            if active_trade is None and consecutive_losses < 5:
                # Get historical slice for analysis
                history = klines[i - min_candles:i + 1]

                signal = self.strategy.analyze(symbol, history, interval)

                if signal and signal.strength >= self.config.strategy.min_signal_strength:
                    # Calculate position size
                    risk_pct = self.risk_config.default_risk_pct
                    if consecutive_losses >= 2:
                        risk_pct *= 0.5  # Reduce size after losses

                    stop_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
                    dollar_risk = balance * (risk_pct / 100)
                    position_value = min(
                        dollar_risk / stop_distance,
                        balance * (self.risk_config.max_position_size_pct / 100)
                    )

                    quantity = position_value / current_price

                    # Apply slippage to entry
                    if signal.signal_type == SignalType.LONG:
                        entry_price = current_price * (1 + self.slippage_pct / 100)
                    else:
                        entry_price = current_price * (1 - self.slippage_pct / 100)

                    trade_id += 1
                    active_trade = BacktestTrade(
                        trade_id=trade_id,
                        symbol=symbol,
                        side=signal.signal_type.value,
                        entry_time=current_time,
                        entry_price=entry_price,
                        quantity=quantity,
                        stop_loss=signal.stop_loss,
                        take_profit_1=signal.take_profit_1,
                        take_profit_2=signal.take_profit_2,
                        take_profit_3=signal.take_profit_3
                    )

                    logger.debug(
                        f"Trade {trade_id}: {signal.signal_type.value} {symbol} @ {entry_price:.4f} | "
                        f"SL: {signal.stop_loss:.4f} | TP1: {signal.take_profit_1:.4f}"
                    )

        # Close any remaining trade at last price
        if active_trade:
            active_trade.exit_price = klines[-1]['close']
            active_trade.exit_time = datetime.fromtimestamp(klines[-1]['timestamp'] / 1000)
            active_trade.exit_reason = "End of backtest"
            self._calculate_pnl(active_trade)
            balance += active_trade.pnl
            trades.append(active_trade)

        # Calculate results
        result = self._calculate_results(
            symbol=symbol,
            klines=klines,
            trades=trades,
            equity_curve=equity_curve,
            max_drawdown=max_drawdown
        )

        return result

    def _check_trade_exit(
        self,
        trade: BacktestTrade,
        high: float,
        low: float,
        close: float,
        current_time: datetime
    ) -> bool:
        """Check if trade should be closed"""

        if trade.side == "LONG":
            # Check stop loss
            if low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss * (1 - self.slippage_pct / 100)
                trade.exit_time = current_time
                trade.exit_reason = "Stop Loss"
                self._calculate_pnl(trade)
                return True

            # Check take profits (scaled exits)
            if not trade.tp1_hit and high >= trade.take_profit_1:
                tp1_qty = trade.quantity * 0.4
                tp1_pnl = (trade.take_profit_1 - trade.entry_price) * tp1_qty
                trade.partial_pnl += tp1_pnl
                trade.tp1_hit = True
                # Move stop to breakeven
                trade.stop_loss = trade.entry_price

            if not trade.tp2_hit and high >= trade.take_profit_2:
                tp2_qty = trade.quantity * 0.3
                tp2_pnl = (trade.take_profit_2 - trade.entry_price) * tp2_qty
                trade.partial_pnl += tp2_pnl
                trade.tp2_hit = True
                # Move stop to TP1
                trade.stop_loss = trade.take_profit_1

            if not trade.tp3_hit and high >= trade.take_profit_3:
                tp3_qty = trade.quantity * 0.3
                tp3_pnl = (trade.take_profit_3 - trade.entry_price) * tp3_qty
                trade.partial_pnl += tp3_pnl
                trade.tp3_hit = True
                trade.exit_price = trade.take_profit_3
                trade.exit_time = current_time
                trade.exit_reason = "TP3 Hit"
                trade.pnl = trade.partial_pnl
                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100
                return True

        else:  # SHORT
            # Check stop loss
            if high >= trade.stop_loss:
                trade.exit_price = trade.stop_loss * (1 + self.slippage_pct / 100)
                trade.exit_time = current_time
                trade.exit_reason = "Stop Loss"
                self._calculate_pnl(trade)
                return True

            # Check take profits
            if not trade.tp1_hit and low <= trade.take_profit_1:
                tp1_qty = trade.quantity * 0.4
                tp1_pnl = (trade.entry_price - trade.take_profit_1) * tp1_qty
                trade.partial_pnl += tp1_pnl
                trade.tp1_hit = True
                trade.stop_loss = trade.entry_price

            if not trade.tp2_hit and low <= trade.take_profit_2:
                tp2_qty = trade.quantity * 0.3
                tp2_pnl = (trade.entry_price - trade.take_profit_2) * tp2_qty
                trade.partial_pnl += tp2_pnl
                trade.tp2_hit = True
                trade.stop_loss = trade.take_profit_1

            if not trade.tp3_hit and low <= trade.take_profit_3:
                tp3_qty = trade.quantity * 0.3
                tp3_pnl = (trade.entry_price - trade.take_profit_3) * tp3_qty
                trade.partial_pnl += tp3_pnl
                trade.tp3_hit = True
                trade.exit_price = trade.take_profit_3
                trade.exit_time = current_time
                trade.exit_reason = "TP3 Hit"
                trade.pnl = trade.partial_pnl
                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100
                return True

        return False

    def _calculate_pnl(self, trade: BacktestTrade):
        """Calculate PnL for a trade"""
        if trade.side == "LONG":
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity

        # Add any partial profits
        trade.pnl += trade.partial_pnl

        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity) * 100

    def _calculate_results(
        self,
        symbol: str,
        klines: List[Dict],
        trades: List[BacktestTrade],
        equity_curve: List[float],
        max_drawdown: float
    ) -> BacktestResult:
        """Calculate comprehensive backtest statistics"""

        if not trades:
            return BacktestResult(
                symbol=symbol,
                start_date=datetime.fromtimestamp(klines[0]['timestamp'] / 1000).strftime('%Y-%m-%d'),
                end_date=datetime.fromtimestamp(klines[-1]['timestamp'] / 1000).strftime('%Y-%m-%d'),
                initial_balance=self.initial_balance,
                final_balance=self.initial_balance,
                trades=[]
            )

        final_balance = self.initial_balance + sum(t.pnl for t in trades)

        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))

        # Calculate average holding time
        holding_times = []
        for t in trades:
            if t.entry_time and t.exit_time:
                delta = t.exit_time - t.entry_time
                holding_times.append(delta.total_seconds() / 3600)  # Hours

        avg_holding_hours = sum(holding_times) / len(holding_times) if holding_times else 0

        result = BacktestResult(
            symbol=symbol,
            start_date=datetime.fromtimestamp(klines[200]['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M'),
            end_date=datetime.fromtimestamp(klines[-1]['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M'),
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            total_pnl=final_balance - self.initial_balance,
            total_pnl_pct=(final_balance - self.initial_balance) / self.initial_balance * 100,
            win_rate=len(winning_trades) / len(trades) * 100 if trades else 0,
            profit_factor=total_wins / total_losses if total_losses > 0 else float('inf'),
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown / self.initial_balance * 100,
            avg_win=total_wins / len(winning_trades) if winning_trades else 0,
            avg_loss=total_losses / len(losing_trades) if losing_trades else 0,
            avg_trade=sum(t.pnl for t in trades) / len(trades) if trades else 0,
            largest_win=max(t.pnl for t in winning_trades) if winning_trades else 0,
            largest_loss=min(t.pnl for t in losing_trades) if losing_trades else 0,
            avg_holding_time=f"{avg_holding_hours:.1f} hours",
            trades=trades,
            equity_curve=equity_curve
        )

        return result

    def print_results(self, result: BacktestResult):
        """Print formatted backtest results"""
        print("\n" + "=" * 60)
        print(f"  BACKTEST RESULTS - {result.symbol}")
        print("=" * 60)
        print(f"\nPeriod: {result.start_date} to {result.end_date}")
        print(f"Initial Balance: ${result.initial_balance:,.2f}")
        print(f"Final Balance:   ${result.final_balance:,.2f}")

        print(f"\n--- Performance ---")
        print(f"Total P&L:       ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)")
        print(f"Max Drawdown:    ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")

        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades:    {result.total_trades}")
        print(f"Winning Trades:  {result.winning_trades}")
        print(f"Losing Trades:   {result.losing_trades}")
        print(f"Win Rate:        {result.win_rate:.1f}%")
        print(f"Profit Factor:   {result.profit_factor:.2f}")

        print(f"\n--- Averages ---")
        print(f"Avg Win:         ${result.avg_win:,.2f}")
        print(f"Avg Loss:        ${result.avg_loss:,.2f}")
        print(f"Avg Trade:       ${result.avg_trade:,.2f}")
        print(f"Avg Holding:     {result.avg_holding_time}")

        print(f"\n--- Extremes ---")
        print(f"Largest Win:     ${result.largest_win:,.2f}")
        print(f"Largest Loss:    ${result.largest_loss:,.2f}")

        print("\n" + "=" * 60)

        # Print recent trades
        if result.trades:
            print("\n--- Recent Trades ---")
            for trade in result.trades[-10:]:
                emoji = "✅" if trade.pnl > 0 else "❌"
                print(
                    f"{emoji} {trade.side:5} | Entry: {trade.entry_price:.4f} | "
                    f"Exit: {trade.exit_price:.4f} | P&L: ${trade.pnl:+.2f} | {trade.exit_reason}"
                )

    def run_multi_symbol_backtest(
        self,
        symbols: List[str],
        interval: str = "15m",
        lookback_candles: int = 1000
    ) -> Dict[str, BacktestResult]:
        """Run backtest across multiple symbols"""
        results = {}

        print(f"\nRunning backtest on {len(symbols)} symbols...")
        print("-" * 40)

        for symbol in symbols:
            print(f"\nBacktesting {symbol}...")
            result = self.run_backtest(symbol, interval, lookback_candles)
            results[symbol] = result

            # Quick summary
            print(f"  Trades: {result.total_trades} | "
                  f"Win Rate: {result.win_rate:.1f}% | "
                  f"P&L: ${result.total_pnl:+,.2f} ({result.total_pnl_pct:+.1f}%)")

        # Overall summary
        print("\n" + "=" * 60)
        print("  OVERALL BACKTEST SUMMARY")
        print("=" * 60)

        total_trades = sum(r.total_trades for r in results.values())
        total_wins = sum(r.winning_trades for r in results.values())
        total_pnl = sum(r.total_pnl for r in results.values())

        print(f"\nSymbols Tested: {len(symbols)}")
        print(f"Total Trades:   {total_trades}")
        print(f"Overall Win Rate: {total_wins / total_trades * 100:.1f}%" if total_trades else "N/A")
        print(f"Total P&L:      ${total_pnl:+,.2f}")

        # Best and worst performers
        if results:
            best = max(results.items(), key=lambda x: x[1].total_pnl_pct)
            worst = min(results.items(), key=lambda x: x[1].total_pnl_pct)

            print(f"\nBest:  {best[0]} ({best[1].total_pnl_pct:+.1f}%)")
            print(f"Worst: {worst[0]} ({worst[1].total_pnl_pct:+.1f}%)")

        print("=" * 60)

        return results


def main():
    """Run backtest from command line"""
    import argparse

    parser = argparse.ArgumentParser(description='Backtest the AMR-VF trading strategy')
    parser.add_argument('--symbol', '-s', default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--interval', '-i', default='15m', help='Timeframe (default: 15m)')
    parser.add_argument('--candles', '-c', type=int, default=1000, help='Number of candles (default: 1000)')
    parser.add_argument('--balance', '-b', type=float, default=10000, help='Initial balance (default: 10000)')
    parser.add_argument('--multi', '-m', action='store_true', help='Test multiple symbols')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    backtester = Backtester(initial_balance=args.balance)

    if args.multi:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
        results = backtester.run_multi_symbol_backtest(symbols, args.interval, args.candles)
    else:
        result = backtester.run_backtest(args.symbol, args.interval, args.candles)
        backtester.print_results(result)


if __name__ == "__main__":
    main()
