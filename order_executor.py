"""
Order Execution Module
Handles trade execution with smart order routing and Algo Orders
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from binance_client import (
    BinanceFuturesClient, OrderSide, OrderType, PositionSide,
    OrderResult
)
from strategy import TradingSignal, SignalType
from risk_manager import Trade, TradeStatus, RiskManager
from config import BotConfig, AlgoOrderConfig, StrategyConfig


logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Order execution mode"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"
    VP = "VP"


@dataclass
class ExecutionPlan:
    """Plan for executing a trade"""
    signal: TradingSignal
    quantity: float
    position_value: float
    execution_mode: ExecutionMode
    leverage: int = 20
    twap_duration: Optional[int] = None


class OrderExecutor:
    """
    Smart Order Execution with Algo Order Support

    Features:
    - Automatic execution mode selection (Market/TWAP/VP)
    - Stop loss and take profit order management
    - Partial close handling
    - Order tracking and reconciliation
    """

    def __init__(
        self,
        client: BinanceFuturesClient,
        risk_manager: RiskManager,
        config: BotConfig
    ):
        self.client = client
        self.risk_manager = risk_manager
        self.config = config
        self.algo_config = config.algo
        self.strategy_config = config.strategy

        # Order tracking
        self.pending_orders: Dict[str, Dict] = {}
        self.filled_orders: Dict[str, Dict] = {}

    def create_execution_plan(
        self,
        signal: TradingSignal,
        account_balance: float
    ) -> Optional[ExecutionPlan]:
        """
        Create an execution plan for a signal

        Determines:
        - Position size
        - Leverage (auto-adjusted for small accounts)
        - Execution mode (Market/TWAP/VP)
        - Order parameters
        """
        # Calculate position size with auto-leverage adjustment
        quantity, position_value, leverage = self.risk_manager.calculate_position_size(
            signal, account_balance, leverage=self.config.default_leverage
        )

        # Validate order
        is_valid, reason = self.client.validate_order(
            signal.symbol, quantity
        )
        if not is_valid:
            logger.error(f"Order validation failed: {reason}")
            return None

        # Determine execution mode
        execution_mode = self._determine_execution_mode(position_value, signal)
        twap_duration = None

        if execution_mode == ExecutionMode.TWAP:
            # Calculate TWAP duration based on position size
            twap_duration = self._calculate_twap_duration(position_value)

        return ExecutionPlan(
            signal=signal,
            quantity=quantity,
            position_value=position_value,
            execution_mode=execution_mode,
            leverage=leverage,
            twap_duration=twap_duration
        )

    def _determine_execution_mode(
        self,
        position_value: float,
        signal: TradingSignal
    ) -> ExecutionMode:
        """Determine best execution mode based on order size"""
        # Use TWAP for large orders
        if (self.algo_config.use_twap_for_large_orders and
            position_value >= self.algo_config.twap_threshold_usdt and
            position_value >= 1000):  # TWAP minimum
            return ExecutionMode.TWAP

        # Use VP for very large orders if enabled
        if (self.algo_config.use_vp_orders and
            position_value >= 10000):
            return ExecutionMode.VP

        return ExecutionMode.MARKET

    def _calculate_twap_duration(self, position_value: float) -> int:
        """Calculate TWAP duration based on order size"""
        # Larger orders get longer duration
        if position_value >= 50000:
            return 1800  # 30 minutes
        elif position_value >= 20000:
            return 900  # 15 minutes
        elif position_value >= 10000:
            return 600  # 10 minutes
        else:
            return 300  # 5 minutes (minimum)

    def execute_trade(
        self,
        plan: ExecutionPlan
    ) -> Tuple[bool, Optional[Trade], str]:
        """
        Execute a trade based on the execution plan

        Returns (success, trade, message)
        """
        signal = plan.signal
        logger.info(
            f"Executing trade: {signal.symbol} {signal.signal_type.value} | "
            f"Mode: {plan.execution_mode.value} | Qty: {plan.quantity:.6f} | "
            f"Leverage: {plan.leverage}x"
        )

        # Set leverage for this symbol
        leverage_set = self.client.set_leverage(signal.symbol, plan.leverage)
        if leverage_set:
            logger.info(f"Leverage set to {plan.leverage}x for {signal.symbol}")
        else:
            logger.warning(f"Could not set leverage to {plan.leverage}x, using current setting")

        # Determine order side
        order_side = OrderSide.BUY if signal.signal_type == SignalType.LONG else OrderSide.SELL
        position_side = PositionSide.BOTH  # One-way mode

        # Execute entry order
        entry_result = self._execute_entry_order(
            symbol=signal.symbol,
            side=order_side,
            quantity=plan.quantity,
            mode=plan.execution_mode,
            twap_duration=plan.twap_duration
        )

        if not entry_result.success:
            return False, None, f"Entry order failed: {entry_result.message}"

        # Create trade record
        trade = self.risk_manager.create_trade(
            signal=signal,
            quantity=plan.quantity,
            entry_order_id=entry_result.order_id
        )
        trade.status = TradeStatus.OPEN
        trade.entry_time = datetime.now()

        # Place stop loss order using Algo Order API
        stop_result = self._place_stop_order(
            trade=trade,
            position_side=position_side
        )

        if stop_result.success:
            trade.stop_order_id = stop_result.order_id
            logger.info(f"Stop loss order placed: {stop_result.order_id}")
        else:
            logger.warning(f"Failed to place stop loss: {stop_result.message}")

        # Place take profit orders
        tp_results = self._place_take_profit_orders(
            trade=trade,
            position_side=position_side
        )

        for level, result in tp_results.items():
            if result.success:
                if level == 'TP1':
                    trade.tp1_order_id = result.order_id
                elif level == 'TP2':
                    trade.tp2_order_id = result.order_id
                elif level == 'TP3':
                    trade.tp3_order_id = result.order_id
                logger.info(f"{level} order placed: {result.order_id}")

        return True, trade, "Trade executed successfully"

    def _execute_entry_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        mode: ExecutionMode,
        twap_duration: Optional[int] = None
    ) -> OrderResult:
        """Execute entry order based on mode"""
        if mode == ExecutionMode.TWAP:
            return self.client.place_twap_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                duration_seconds=twap_duration or 600
            )
        elif mode == ExecutionMode.VP:
            return self.client.place_vp_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                urgency=self.algo_config.vp_urgency
            )
        else:
            return self.client.execute_smart_order(
                symbol=symbol,
                side=side,
                quantity=quantity
            )

    def _place_stop_order(
        self,
        trade: Trade,
        position_side: PositionSide
    ) -> OrderResult:
        """
        Place stop loss order using Algo Order API

        Uses the new /fapi/v1/algoOrder endpoint for conditional orders
        """
        # Determine stop side (opposite of position)
        stop_side = OrderSide.SELL if trade.side == 'LONG' else OrderSide.BUY

        return self.client.place_order(
            symbol=trade.symbol,
            side=stop_side,
            quantity=trade.quantity,
            order_type=OrderType.STOP_MARKET,
            stop_price=trade.stop_loss,
            position_side=position_side,
            reduce_only=True
        )

    def _place_take_profit_orders(
        self,
        trade: Trade,
        position_side: PositionSide
    ) -> Dict[str, OrderResult]:
        """Place multiple take profit orders for scaled exits"""
        results = {}
        close_side = OrderSide.SELL if trade.side == 'LONG' else OrderSide.BUY

        # Calculate quantities for each TP level
        tp1_qty = trade.quantity * (self.strategy_config.tp1_size_pct / 100)
        tp2_qty = trade.quantity * (self.strategy_config.tp2_size_pct / 100)
        tp3_qty = trade.quantity * (self.strategy_config.tp3_size_pct / 100)

        # TP1
        results['TP1'] = self.client.place_order(
            symbol=trade.symbol,
            side=close_side,
            quantity=tp1_qty,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            stop_price=trade.take_profit_1,
            position_side=position_side,
            reduce_only=True
        )

        # TP2
        results['TP2'] = self.client.place_order(
            symbol=trade.symbol,
            side=close_side,
            quantity=tp2_qty,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            stop_price=trade.take_profit_2,
            position_side=position_side,
            reduce_only=True
        )

        # TP3
        results['TP3'] = self.client.place_order(
            symbol=trade.symbol,
            side=close_side,
            quantity=tp3_qty,
            order_type=OrderType.TAKE_PROFIT_MARKET,
            stop_price=trade.take_profit_3,
            position_side=position_side,
            reduce_only=True
        )

        return results

    def close_trade(
        self,
        trade: Trade,
        reason: str = "Manual close"
    ) -> Tuple[bool, str]:
        """Close an open trade immediately"""
        logger.info(f"Closing trade {trade.trade_id}: {reason}")

        # Cancel existing stop/TP orders
        self._cancel_trade_orders(trade)

        # Place market close order
        close_side = OrderSide.SELL if trade.side == 'LONG' else OrderSide.BUY

        result = self.client.place_order(
            symbol=trade.symbol,
            side=close_side,
            quantity=trade.remaining_quantity,
            order_type=OrderType.MARKET,
            reduce_only=True
        )

        if result.success:
            # Get exit price from order result
            exit_price = result.raw_response.get('avgPrice', trade.entry_price)

            # Calculate PnL
            if trade.side == 'LONG':
                pnl = (exit_price - trade.entry_price) * trade.remaining_quantity
            else:
                pnl = (trade.entry_price - exit_price) * trade.remaining_quantity

            self.risk_manager.update_trade_status(
                trade.trade_id,
                TradeStatus.CLOSED,
                exit_price=exit_price,
                realized_pnl=pnl
            )

            return True, f"Trade closed. PnL: ${pnl:.2f}"

        return False, f"Failed to close trade: {result.message}"

    def _cancel_trade_orders(self, trade: Trade):
        """Cancel all pending orders for a trade"""
        orders_to_cancel = [
            trade.stop_order_id,
            trade.tp1_order_id,
            trade.tp2_order_id,
            trade.tp3_order_id
        ]

        for order_id in orders_to_cancel:
            if order_id:
                try:
                    self.client.cancel_algo_order(order_id)
                    logger.debug(f"Cancelled order: {order_id}")
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order_id}: {e}")

    def update_trailing_stop(
        self,
        trade: Trade,
        new_stop_price: float
    ) -> bool:
        """Update trailing stop price for a trade"""
        logger.info(
            f"Updating trailing stop for {trade.trade_id}: "
            f"{trade.stop_loss:.4f} -> {new_stop_price:.4f}"
        )

        # Cancel existing stop order
        if trade.stop_order_id:
            self.client.cancel_algo_order(trade.stop_order_id)

        # Place new stop order
        stop_side = OrderSide.SELL if trade.side == 'LONG' else OrderSide.BUY

        result = self.client.place_order(
            symbol=trade.symbol,
            side=stop_side,
            quantity=trade.remaining_quantity,
            order_type=OrderType.STOP_MARKET,
            stop_price=new_stop_price,
            reduce_only=True
        )

        if result.success:
            trade.stop_loss = new_stop_price
            trade.stop_order_id = result.order_id
            return True

        return False

    def execute_partial_close(
        self,
        trade: Trade,
        quantity: float,
        reason: str
    ) -> Tuple[bool, float]:
        """
        Execute a partial close for take profit

        Returns (success, realized_pnl)
        """
        logger.info(f"Partial close for {trade.trade_id}: {quantity:.6f} ({reason})")

        close_side = OrderSide.SELL if trade.side == 'LONG' else OrderSide.BUY

        result = self.client.place_order(
            symbol=trade.symbol,
            side=close_side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            reduce_only=True
        )

        if result.success:
            exit_price = float(result.raw_response.get('avgPrice', 0))

            # Calculate PnL for this portion
            if trade.side == 'LONG':
                pnl = (exit_price - trade.entry_price) * quantity
            else:
                pnl = (trade.entry_price - exit_price) * quantity

            # Update remaining quantity
            trade.remaining_quantity -= quantity
            trade.realized_pnl += pnl

            # Update status if fully closed
            if trade.remaining_quantity <= 0:
                self.risk_manager.update_trade_status(
                    trade.trade_id,
                    TradeStatus.CLOSED,
                    exit_price=exit_price,
                    realized_pnl=trade.realized_pnl
                )
            else:
                trade.status = TradeStatus.PARTIAL_CLOSE

            return True, pnl

        return False, 0.0

    def sync_positions(self) -> Dict[str, Dict]:
        """
        Sync local trade state with actual positions on exchange

        Returns dict of symbol -> position info
        """
        positions = self.client.get_positions()
        synced = {}

        for pos in positions:
            synced[pos.symbol] = {
                'size': pos.size,
                'side': pos.side,
                'entry_price': pos.entry_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'leverage': pos.leverage,
                'liquidation_price': pos.liquidation_price
            }

            # Update trade unrealized PnL if we have matching trade
            for trade in self.risk_manager.active_trades.values():
                if (trade.symbol == pos.symbol and
                    trade.side == pos.side and
                    trade.status == TradeStatus.OPEN):
                    trade.unrealized_pnl = pos.unrealized_pnl

        return synced

    def check_algo_order_status(self, algo_id: str) -> Dict:
        """Check status of an algo order"""
        return self.client.get_algo_order(algo_id=algo_id)

    def get_open_algo_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open algo orders"""
        return self.client.get_open_algo_orders(symbol)


class PaperTradeExecutor:
    """
    Paper trading executor for testing strategies without real orders

    Simulates order execution with realistic fills
    """

    def __init__(
        self,
        client: BinanceFuturesClient,
        risk_manager: RiskManager,
        config: BotConfig,
        initial_balance: float = 10000.0
    ):
        self.client = client
        self.risk_manager = risk_manager
        self.config = config
        self.balance = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.order_counter = 0

        # Slippage simulation
        self.slippage_pct = 0.05  # 0.05% slippage
        self.fee_pct = 0.04  # 0.04% taker fee

    def execute_trade(
        self,
        plan: ExecutionPlan
    ) -> Tuple[bool, Optional[Trade], str]:
        """Execute paper trade"""
        signal = plan.signal

        # Apply slippage
        if signal.signal_type == SignalType.LONG:
            fill_price = signal.entry_price * (1 + self.slippage_pct / 100)
        else:
            fill_price = signal.entry_price * (1 - self.slippage_pct / 100)

        # Calculate fees
        position_value = plan.quantity * fill_price
        fees = position_value * (self.fee_pct / 100)

        # Check if we have enough balance
        margin_required = position_value / self.config.default_leverage
        if margin_required + fees > self.balance:
            return False, None, "Insufficient balance"

        # Generate order ID
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}_{int(time.time() * 1000)}"

        # Create trade
        trade = self.risk_manager.create_trade(
            signal=signal,
            quantity=plan.quantity,
            entry_order_id=order_id
        )
        trade.status = TradeStatus.OPEN
        trade.entry_time = datetime.now()
        trade.entry_price = fill_price
        trade.fees = fees

        # Update balance
        self.balance -= fees

        # Store position
        self.positions[signal.symbol] = {
            'trade': trade,
            'entry_price': fill_price,
            'quantity': plan.quantity,
            'side': signal.signal_type.value
        }

        logger.info(
            f"[PAPER] Trade executed: {signal.symbol} {signal.signal_type.value} | "
            f"Fill: {fill_price:.4f} | Fee: ${fees:.2f}"
        )

        return True, trade, "Paper trade executed"

    def update_positions(self, prices: Dict[str, float]):
        """Update paper positions with current prices"""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                trade = pos['trade']

                # Calculate unrealized PnL
                if pos['side'] == 'LONG':
                    pnl = (current_price - pos['entry_price']) * pos['quantity']
                else:
                    pnl = (pos['entry_price'] - current_price) * pos['quantity']

                trade.unrealized_pnl = pnl

                # Check stop loss
                if pos['side'] == 'LONG' and current_price <= trade.stop_loss:
                    self._close_paper_position(symbol, current_price, 'Stop Loss')
                elif pos['side'] == 'SHORT' and current_price >= trade.stop_loss:
                    self._close_paper_position(symbol, current_price, 'Stop Loss')

                # Check take profits
                tps = self.risk_manager.check_partial_take_profits(trade, current_price)
                for tp_level, qty in tps:
                    self._partial_close_paper(symbol, current_price, qty, tp_level)

    def _close_paper_position(
        self,
        symbol: str,
        price: float,
        reason: str
    ):
        """Close paper position"""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        trade = pos['trade']

        # Apply slippage
        if pos['side'] == 'LONG':
            fill_price = price * (1 - self.slippage_pct / 100)
            pnl = (fill_price - pos['entry_price']) * pos['quantity']
        else:
            fill_price = price * (1 + self.slippage_pct / 100)
            pnl = (pos['entry_price'] - fill_price) * pos['quantity']

        # Subtract fees
        fees = pos['quantity'] * fill_price * (self.fee_pct / 100)
        pnl -= fees

        # Update balance
        self.balance += pnl

        # Update trade
        status = TradeStatus.STOPPED if 'Stop' in reason else TradeStatus.CLOSED
        self.risk_manager.update_trade_status(
            trade.trade_id,
            status,
            exit_price=fill_price,
            realized_pnl=pnl
        )

        del self.positions[symbol]

        logger.info(
            f"[PAPER] Position closed: {symbol} | {reason} | "
            f"Exit: {fill_price:.4f} | PnL: ${pnl:.2f}"
        )

    def _partial_close_paper(
        self,
        symbol: str,
        price: float,
        quantity: float,
        reason: str
    ):
        """Partial close paper position"""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        trade = pos['trade']

        # Calculate PnL for portion
        if pos['side'] == 'LONG':
            fill_price = price * (1 - self.slippage_pct / 100)
            pnl = (fill_price - pos['entry_price']) * quantity
        else:
            fill_price = price * (1 + self.slippage_pct / 100)
            pnl = (pos['entry_price'] - fill_price) * quantity

        fees = quantity * fill_price * (self.fee_pct / 100)
        pnl -= fees

        # Update balance and position
        self.balance += pnl
        pos['quantity'] -= quantity
        trade.remaining_quantity -= quantity
        trade.realized_pnl += pnl

        logger.info(
            f"[PAPER] Partial close: {symbol} | {reason} | "
            f"Qty: {quantity:.6f} | PnL: ${pnl:.2f}"
        )

        # Remove if fully closed
        if pos['quantity'] <= 0:
            del self.positions[symbol]
            self.risk_manager.update_trade_status(
                trade.trade_id,
                TradeStatus.CLOSED,
                exit_price=fill_price,
                realized_pnl=trade.realized_pnl
            )

    def get_balance(self) -> float:
        """Get current paper balance"""
        return self.balance

    def get_equity(self) -> float:
        """Get total equity (balance + unrealized PnL)"""
        unrealized = sum(
            pos['trade'].unrealized_pnl
            for pos in self.positions.values()
        )
        return self.balance + unrealized
