"""
Risk Management System
Handles position sizing, drawdown protection, and trade management
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import json

from config import RiskConfig, BotConfig
from strategy import TradingSignal, SignalType


logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Trade status enum"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"
    CANCELLED = "CANCELLED"


@dataclass
class Trade:
    """Represents a trade with full tracking"""
    trade_id: str
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float

    status: TradeStatus = TradeStatus.PENDING
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees: float = 0.0

    # Partial closes
    remaining_quantity: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False

    # Orders
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    tp1_order_id: Optional[str] = None
    tp2_order_id: Optional[str] = None
    tp3_order_id: Optional[str] = None

    # Signal info
    signal_strength: int = 0
    confluence_factors: int = 0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'take_profit_3': self.take_profit_3,
            'status': self.status.value,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'fees': self.fees,
            'remaining_quantity': self.remaining_quantity,
            'signal_strength': self.signal_strength,
            'confluence_factors': self.confluence_factors,
            'reasons': self.reasons
        }


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: str
    starting_balance: float
    current_balance: float
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0


class RiskManager:
    """
    Comprehensive Risk Management System

    Features:
    - Dynamic position sizing based on volatility and win streak
    - Daily drawdown protection
    - Consecutive loss handling
    - Maximum exposure limits
    - Trade tracking and statistics
    """

    def __init__(self, config: RiskConfig, db_path: str = "trading_data.db"):
        self.config = config
        self.db_path = db_path

        # Active trades
        self.active_trades: Dict[str, Trade] = {}

        # Daily statistics
        self.daily_stats: Optional[DailyStats] = None
        self.starting_balance: float = 0.0
        self.current_balance: float = 0.0
        self.peak_balance: float = 0.0

        # Loss tracking
        self.consecutive_losses: int = 0
        self.last_loss_time: Optional[datetime] = None
        self.in_cooldown: bool = False

        # Daily counters
        self.trades_today: int = 0
        self.daily_pnl: float = 0.0

        # Initialize database
        self._init_database()
        self._load_state()

    def _init_database(self):
        """Initialize SQLite database for trade history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                quantity REAL,
                stop_loss REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                take_profit_3 REAL,
                status TEXT,
                entry_time TEXT,
                exit_time TEXT,
                exit_price REAL,
                realized_pnl REAL,
                fees REAL,
                signal_strength INTEGER,
                confluence_factors INTEGER,
                reasons TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                starting_balance REAL,
                ending_balance REAL,
                trades_count INTEGER,
                wins INTEGER,
                losses INTEGER,
                total_pnl REAL,
                max_drawdown REAL,
                peak_balance REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _load_state(self):
        """Load persistent state from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load consecutive losses
        cursor.execute("SELECT value FROM bot_state WHERE key = 'consecutive_losses'")
        row = cursor.fetchone()
        if row:
            self.consecutive_losses = int(row[0])

        # Load today's stats
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (today,))
        row = cursor.fetchone()
        if row:
            self.trades_today = row[3] or 0
            self.daily_pnl = row[6] or 0.0

        conn.close()

    def _save_state(self):
        """Save state to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO bot_state (key, value, updated_at)
            VALUES ('consecutive_losses', ?, CURRENT_TIMESTAMP)
        ''', (str(self.consecutive_losses),))

        conn.commit()
        conn.close()

    def update_balance(self, balance: float):
        """Update current balance"""
        self.current_balance = balance

        if self.starting_balance == 0:
            self.starting_balance = balance
            self.peak_balance = balance

        if balance > self.peak_balance:
            self.peak_balance = balance

    def calculate_position_size(
        self,
        signal: TradingSignal,
        account_balance: float,
        leverage: int = 20,
        min_notional: float = 5.0
    ) -> Tuple[float, float, int]:
        """
        Calculate position size based on risk parameters
        Automatically adjusts leverage for small accounts to meet minimum notional

        Returns (quantity in base asset, position value in USDT, required leverage)
        """
        # Minimum notional for Binance Futures (usually $5-10)
        MIN_NOTIONAL = max(min_notional, 5.0)
        MAX_LEVERAGE = 125  # Binance max for most pairs

        # Base risk percentage - use more of balance for small accounts
        if account_balance < 10:
            risk_pct = 80.0  # Use 80% of tiny accounts
        elif account_balance < 50:
            risk_pct = 50.0  # Use 50% of small accounts
        else:
            risk_pct = self.config.default_risk_pct

        # Adjust for consecutive losses
        if self.config.reduce_size_after_losses and self.consecutive_losses >= 2:
            risk_pct *= self.config.size_reduction_factor
            logger.info(f"Reduced position size due to {self.consecutive_losses} consecutive losses")

        # Adjust for signal strength (higher strength = slightly larger position)
        strength_multiplier = 0.8 + (signal.strength / 10) * 0.4  # 0.8x to 1.2x
        risk_pct *= strength_multiplier

        # Cap at maximum risk
        risk_pct = min(risk_pct, self.config.max_risk_pct)

        # Calculate dollar risk
        dollar_risk = account_balance * (risk_pct / 100)

        # Calculate position size based on stop loss distance
        stop_distance_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        position_value = dollar_risk / stop_distance_pct if stop_distance_pct > 0 else dollar_risk

        # Cap at maximum position size based on balance
        max_position = account_balance * (self.config.max_position_size_pct / 100)
        position_value = min(position_value, max_position)

        # Ensure minimum notional is met
        if position_value < MIN_NOTIONAL:
            position_value = MIN_NOTIONAL
            logger.info(f"Adjusted position to minimum notional: ${MIN_NOTIONAL}")

        # Calculate required leverage
        margin_available = account_balance * 0.95  # Keep 5% buffer
        required_leverage = int(position_value / margin_available) + 1
        required_leverage = max(leverage, required_leverage)
        required_leverage = min(required_leverage, MAX_LEVERAGE)

        # Final margin check - can we afford this position?
        margin_required = position_value / required_leverage
        if margin_required > account_balance:
            # Reduce position to fit available margin
            position_value = account_balance * required_leverage * 0.9  # 90% of max

        # Calculate quantity
        quantity = position_value / signal.entry_price

        logger.info(
            f"Position sizing: Balance ${account_balance:.2f} | "
            f"Position: ${position_value:.2f} | Leverage: {required_leverage}x | "
            f"Quantity: {quantity:.6f} | Margin: ${position_value/required_leverage:.2f}"
        )

        return quantity, position_value, required_leverage

    def can_open_trade(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Check if a new trade can be opened based on risk rules

        Returns (can_trade, reason)
        """
        # Check cooldown
        if self.in_cooldown:
            if self.last_loss_time:
                cooldown_end = self.last_loss_time + timedelta(
                    minutes=self.config.cooldown_after_max_losses_minutes
                )
                if datetime.now() < cooldown_end:
                    remaining = (cooldown_end - datetime.now()).seconds // 60
                    return False, f"In cooldown period ({remaining} minutes remaining)"
                else:
                    self.in_cooldown = False

        # Check daily trade limit
        if self.trades_today >= self.config.max_trades_per_day:
            return False, f"Daily trade limit reached ({self.config.max_trades_per_day})"

        # Check concurrent positions
        if len(self.active_trades) >= self.config.max_concurrent_positions:
            return False, f"Max concurrent positions reached ({self.config.max_concurrent_positions})"

        # Check if already have position in this symbol
        for trade in self.active_trades.values():
            if trade.symbol == signal.symbol and trade.status == TradeStatus.OPEN:
                return False, f"Already have open position in {signal.symbol}"

        # Check daily drawdown
        if self.current_balance > 0 and self.starting_balance > 0:
            daily_drawdown = (self.starting_balance - self.current_balance) / self.starting_balance * 100
            if daily_drawdown >= self.config.max_daily_loss_pct:
                return False, f"Daily drawdown limit reached ({daily_drawdown:.2f}%)"

        # Check total exposure
        total_exposure = sum(
            trade.quantity * trade.entry_price
            for trade in self.active_trades.values()
            if trade.status == TradeStatus.OPEN
        )
        max_exposure = self.current_balance * (self.config.max_total_exposure_pct / 100)
        if total_exposure >= max_exposure:
            return False, f"Maximum total exposure reached (${total_exposure:.2f})"

        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.in_cooldown = True
            self.last_loss_time = datetime.now()
            return False, f"Max consecutive losses reached ({self.consecutive_losses})"

        # Check risk/reward
        if signal.risk_reward_ratio < self.config.min_risk_reward:
            return False, f"R:R too low ({signal.risk_reward_ratio:.2f} < {self.config.min_risk_reward})"

        return True, "OK"

    def create_trade(
        self,
        signal: TradingSignal,
        quantity: float,
        entry_order_id: Optional[str] = None
    ) -> Trade:
        """Create a new trade from signal"""
        trade_id = f"{signal.symbol}_{int(time.time() * 1000)}"

        trade = Trade(
            trade_id=trade_id,
            symbol=signal.symbol,
            side=signal.signal_type.value,
            entry_price=signal.entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=signal.take_profit_3,
            remaining_quantity=quantity,
            entry_order_id=entry_order_id,
            signal_strength=signal.strength,
            confluence_factors=signal.confluence_factors,
            reasons=signal.reasons.copy()
        )

        self.active_trades[trade_id] = trade
        self.trades_today += 1

        logger.info(f"Trade created: {trade_id} | {signal.symbol} {signal.signal_type.value}")

        return trade

    def update_trade_status(
        self,
        trade_id: str,
        status: TradeStatus,
        exit_price: Optional[float] = None,
        realized_pnl: Optional[float] = None
    ):
        """Update trade status"""
        if trade_id not in self.active_trades:
            logger.warning(f"Trade {trade_id} not found")
            return

        trade = self.active_trades[trade_id]
        trade.status = status

        if status in [TradeStatus.CLOSED, TradeStatus.STOPPED]:
            trade.exit_time = datetime.now()
            trade.exit_price = exit_price

            if realized_pnl is not None:
                trade.realized_pnl = realized_pnl
                self.daily_pnl += realized_pnl

                # Update consecutive losses
                if realized_pnl < 0:
                    self.consecutive_losses += 1
                    logger.info(f"Loss recorded. Consecutive losses: {self.consecutive_losses}")
                else:
                    self.consecutive_losses = 0
                    logger.info("Win recorded. Consecutive losses reset to 0")

                self._save_state()

            # Save to database
            self._save_trade(trade)

            # Remove from active trades
            del self.active_trades[trade_id]

            logger.info(
                f"Trade closed: {trade_id} | PnL: ${realized_pnl:.2f if realized_pnl else 0}"
            )

    def _save_trade(self, trade: Trade):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO trades (
                trade_id, symbol, side, entry_price, quantity,
                stop_loss, take_profit_1, take_profit_2, take_profit_3,
                status, entry_time, exit_time, exit_price,
                realized_pnl, fees, signal_strength, confluence_factors, reasons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.trade_id, trade.symbol, trade.side, trade.entry_price, trade.quantity,
            trade.stop_loss, trade.take_profit_1, trade.take_profit_2, trade.take_profit_3,
            trade.status.value,
            trade.entry_time.isoformat() if trade.entry_time else None,
            trade.exit_time.isoformat() if trade.exit_time else None,
            trade.exit_price, trade.realized_pnl, trade.fees,
            trade.signal_strength, trade.confluence_factors,
            json.dumps(trade.reasons)
        ))

        conn.commit()
        conn.close()

    def update_daily_stats(self, ending_balance: float):
        """Update and save daily statistics"""
        today = datetime.now().strftime('%Y-%m-%d')

        # Calculate stats
        wins = losses = 0
        for trade in self.get_today_trades():
            if trade.realized_pnl > 0:
                wins += 1
            elif trade.realized_pnl < 0:
                losses += 1

        max_dd = 0
        if self.peak_balance > 0:
            max_dd = (self.peak_balance - min(self.current_balance, self.peak_balance)) / self.peak_balance * 100

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO daily_stats (
                date, starting_balance, ending_balance, trades_count,
                wins, losses, total_pnl, max_drawdown, peak_balance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            today, self.starting_balance, ending_balance, self.trades_today,
            wins, losses, self.daily_pnl, max_dd, self.peak_balance
        ))

        conn.commit()
        conn.close()

    def get_today_trades(self) -> List[Trade]:
        """Get all trades from today"""
        today = datetime.now().strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM trades
            WHERE date(entry_time) = ?
        ''', (today,))

        trades = []
        for row in cursor.fetchall():
            trades.append(self._row_to_trade(row))

        conn.close()
        return trades

    def _row_to_trade(self, row) -> Trade:
        """Convert database row to Trade object"""
        return Trade(
            trade_id=row[0],
            symbol=row[1],
            side=row[2],
            entry_price=row[3],
            quantity=row[4],
            stop_loss=row[5],
            take_profit_1=row[6],
            take_profit_2=row[7],
            take_profit_3=row[8],
            status=TradeStatus(row[9]),
            entry_time=datetime.fromisoformat(row[10]) if row[10] else None,
            exit_time=datetime.fromisoformat(row[11]) if row[11] else None,
            exit_price=row[12],
            realized_pnl=row[13] or 0,
            fees=row[14] or 0,
            signal_strength=row[15] or 0,
            confluence_factors=row[16] or 0,
            reasons=json.loads(row[17]) if row[17] else []
        )

    def get_statistics(self, days: int = 30) -> Dict:
        """Get trading statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT * FROM trades
            WHERE date(entry_time) >= ?
            AND status IN ('CLOSED', 'STOPPED')
        ''', (cutoff_date,))

        trades = [self._row_to_trade(row) for row in cursor.fetchall()]
        conn.close()

        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_rr_achieved': 0,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            }

        wins = [t for t in trades if t.realized_pnl > 0]
        losses = [t for t in trades if t.realized_pnl < 0]

        total_wins = sum(t.realized_pnl for t in wins)
        total_losses = abs(sum(t.realized_pnl for t in losses))

        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'total_pnl': sum(t.realized_pnl for t in trades),
            'avg_win': total_wins / len(wins) if wins else 0,
            'avg_loss': total_losses / len(losses) if losses else 0,
            'largest_win': max(t.realized_pnl for t in wins) if wins else 0,
            'largest_loss': min(t.realized_pnl for t in losses) if losses else 0,
            'expectancy': (
                (len(wins) / len(trades) * (total_wins / len(wins) if wins else 0)) -
                (len(losses) / len(trades) * (total_losses / len(losses) if losses else 0))
            ) if trades else 0
        }

    def get_active_trades_summary(self) -> Dict:
        """Get summary of active trades"""
        total_exposure = 0
        total_unrealized_pnl = 0

        for trade in self.active_trades.values():
            if trade.status == TradeStatus.OPEN:
                total_exposure += trade.quantity * trade.entry_price
                total_unrealized_pnl += trade.unrealized_pnl

        return {
            'active_trades': len(self.active_trades),
            'total_exposure': total_exposure,
            'total_unrealized_pnl': total_unrealized_pnl,
            'trades': [t.to_dict() for t in self.active_trades.values()]
        }

    def reset_daily_counters(self):
        """Reset daily counters (call at start of trading day)"""
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.starting_balance = self.current_balance
        self.peak_balance = self.current_balance

        logger.info("Daily counters reset")

    def calculate_trailing_stop(
        self,
        trade: Trade,
        current_price: float
    ) -> Optional[float]:
        """
        Calculate new trailing stop level

        Returns new stop price if it should be updated, None otherwise
        """
        if not self.config.use_trailing_stop:
            return None

        entry = trade.entry_price
        current_stop = trade.stop_loss

        if trade.side == 'LONG':
            # Check if in profit enough to activate trailing stop
            profit_pct = (current_price - entry) / entry * 100
            if profit_pct < self.config.trailing_stop_activation_pct:
                return None

            # Calculate new stop
            new_stop = current_price * (1 - self.config.trailing_stop_callback_pct / 100)

            # Only update if new stop is higher
            if new_stop > current_stop:
                return new_stop

        elif trade.side == 'SHORT':
            profit_pct = (entry - current_price) / entry * 100
            if profit_pct < self.config.trailing_stop_activation_pct:
                return None

            new_stop = current_price * (1 + self.config.trailing_stop_callback_pct / 100)

            if new_stop < current_stop:
                return new_stop

        return None

    def check_partial_take_profits(
        self,
        trade: Trade,
        current_price: float
    ) -> List[Tuple[str, float]]:
        """
        Check if any take profit levels are hit

        Returns list of (tp_level, quantity_to_close)
        """
        closes = []

        if trade.side == 'LONG':
            if not trade.tp1_hit and current_price >= trade.take_profit_1:
                qty = trade.quantity * (self.config.tp1_size_pct / 100)
                closes.append(('TP1', qty))
                trade.tp1_hit = True

            if not trade.tp2_hit and current_price >= trade.take_profit_2:
                qty = trade.quantity * (self.config.tp2_size_pct / 100)
                closes.append(('TP2', qty))
                trade.tp2_hit = True

            if not trade.tp3_hit and current_price >= trade.take_profit_3:
                qty = trade.remaining_quantity
                closes.append(('TP3', qty))
                trade.tp3_hit = True

        elif trade.side == 'SHORT':
            if not trade.tp1_hit and current_price <= trade.take_profit_1:
                qty = trade.quantity * (self.config.tp1_size_pct / 100)
                closes.append(('TP1', qty))
                trade.tp1_hit = True

            if not trade.tp2_hit and current_price <= trade.take_profit_2:
                qty = trade.quantity * (self.config.tp2_size_pct / 100)
                closes.append(('TP2', qty))
                trade.tp2_hit = True

            if not trade.tp3_hit and current_price <= trade.take_profit_3:
                qty = trade.remaining_quantity
                closes.append(('TP3', qty))
                trade.tp3_hit = True

        return closes
