"""
Binance Futures API Client with Algo Order Support
Implements connection to Binance USDT-M Futures with the new Algo Order endpoints
"""

import hashlib
import hmac
import time
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlencode
from collections import deque
import threading

from config import APIConfig, AlgoOrderConfig


logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class PositionSide(Enum):
    BOTH = "BOTH"
    LONG = "LONG"
    SHORT = "SHORT"


class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTX = "GTX"  # Good Till Crossing


class AlgoOrderType(Enum):
    TWAP = "TWAP"
    VP = "VP"


@dataclass
class SymbolInfo:
    """Symbol trading information"""
    symbol: str
    price_precision: int
    quantity_precision: int
    min_qty: float
    max_qty: float
    step_size: float
    tick_size: float
    min_notional: float


@dataclass
class Position:
    """Position information"""
    symbol: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    margin_type: str
    liquidation_price: float


@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[str]
    client_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    message: str
    raw_response: Dict


class RateLimiter:
    """Rate limiter to prevent API throttling"""

    def __init__(self, max_requests: int = 10, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove old requests
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()

            # Wait if we've hit the limit
            if len(self.requests) >= self.max_requests:
                sleep_time = self.time_window - (now - self.requests[0]) + 0.1
                if sleep_time > 0:
                    logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)

                # Clean up after sleeping
                now = time.time()
                while self.requests and self.requests[0] < now - self.time_window:
                    self.requests.popleft()

            self.requests.append(now)


class BinanceFuturesClient:
    """
    Binance USDT-M Futures API Client with Algo Order Support

    Implements the new algo order endpoints:
    - POST /fapi/v1/algoOrder - New algo order (conditional orders)
    - POST /sapi/v1/algo/futures/newOrderTwap - TWAP orders
    - POST /sapi/v1/algo/futures/newOrderVp - VP orders
    """

    def __init__(self, config: APIConfig, algo_config: Optional[AlgoOrderConfig] = None):
        self.config = config
        self.algo_config = algo_config or AlgoOrderConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-MBX-APIKEY': config.api_key
        })
        self.rate_limiter = RateLimiter(max_requests=10, time_window=1)
        self.symbol_info_cache: Dict[str, SymbolInfo] = {}
        self._lock = threading.Lock()

    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for request"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
        use_sapi: bool = False,
        max_retries: int = 3
    ) -> Dict:
        """Make HTTP request to Binance API"""
        params = params or {}

        # Determine base URL
        if use_sapi:
            # SAPI endpoints use the main API domain
            base_url = "https://api.binance.com" if not self.config.testnet else "https://testnet.binance.vision"
        else:
            base_url = self.config.base_url

        url = f"{base_url}{endpoint}"

        # Add timestamp and signature for signed requests
        if signed:
            params['timestamp'] = self._get_timestamp()
            params['recvWindow'] = 5000
            params['signature'] = self._generate_signature(params)

        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()

                if method.upper() == 'GET':
                    response = self.session.get(url, params=params, timeout=30)
                elif method.upper() == 'POST':
                    response = self.session.post(url, data=params, timeout=30)
                elif method.upper() == 'DELETE':
                    response = self.session.delete(url, params=params, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Log response for debugging
                logger.debug(f"API Response [{response.status_code}]: {response.text[:500]}")

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit hit
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit hit, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                else:
                    error_data = response.json() if response.text else {}
                    error_code = error_data.get('code', response.status_code)
                    error_msg = error_data.get('msg', response.text)
                    logger.error(f"API Error {error_code}: {error_msg}")

                    # Don't retry on client errors (except rate limit)
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        return {'error': True, 'code': error_code, 'msg': error_msg}

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue

        return {'error': True, 'code': -1, 'msg': 'Max retries exceeded'}

    # ==================== Market Data ====================

    def get_exchange_info(self) -> Dict:
        """Get exchange information including trading rules"""
        return self._make_request('GET', self.config.exchange_info_endpoint)

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get trading info for a specific symbol"""
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]

        exchange_info = self.get_exchange_info()
        if 'error' in exchange_info:
            return None

        for sym_info in exchange_info.get('symbols', []):
            if sym_info['symbol'] == symbol:
                filters = {f['filterType']: f for f in sym_info.get('filters', [])}

                lot_filter = filters.get('LOT_SIZE', {})
                price_filter = filters.get('PRICE_FILTER', {})
                notional_filter = filters.get('MIN_NOTIONAL', {})

                info = SymbolInfo(
                    symbol=symbol,
                    price_precision=sym_info.get('pricePrecision', 2),
                    quantity_precision=sym_info.get('quantityPrecision', 3),
                    min_qty=float(lot_filter.get('minQty', 0.001)),
                    max_qty=float(lot_filter.get('maxQty', 9999999)),
                    step_size=float(lot_filter.get('stepSize', 0.001)),
                    tick_size=float(price_filter.get('tickSize', 0.01)),
                    min_notional=float(notional_filter.get('notional', 5))
                )
                self.symbol_info_cache[symbol] = info
                return info

        return None

    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        result = self._make_request('GET', self.config.ticker_endpoint, {'symbol': symbol})
        if 'error' not in result:
            return float(result.get('price', 0))
        return None

    def get_klines(
        self,
        symbol: str,
        interval: str = '15m',
        limit: int = 200
    ) -> Optional[List[Dict]]:
        """Get candlestick data"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        result = self._make_request('GET', self.config.klines_endpoint, params)

        if 'error' in result or not isinstance(result, list):
            return None

        klines = []
        for candle in result:
            klines.append({
                'timestamp': int(candle[0]),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),
                'close_time': int(candle[6]),
                'quote_volume': float(candle[7]),
                'trades': int(candle[8]),
                'taker_buy_volume': float(candle[9]),
                'taker_buy_quote_volume': float(candle[10])
            })

        return klines

    def get_all_tickers(self) -> Dict[str, float]:
        """Get all ticker prices"""
        result = self._make_request('GET', self.config.ticker_endpoint)
        if 'error' in result or not isinstance(result, list):
            return {}
        return {item['symbol']: float(item['price']) for item in result}

    # ==================== Account Data ====================

    def get_account_info(self) -> Dict:
        """Get account information including balance"""
        return self._make_request('GET', self.config.account_endpoint, signed=True)

    def get_balance(self, asset: str = 'USDT') -> Optional[float]:
        """Get available balance for an asset"""
        account = self.get_account_info()
        if 'error' in account:
            return None

        for balance in account.get('assets', []):
            if balance['asset'] == asset:
                return float(balance.get('availableBalance', 0))
        return None

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get current positions"""
        params = {}
        if symbol:
            params['symbol'] = symbol

        result = self._make_request('GET', self.config.position_endpoint, params, signed=True)

        if 'error' in result or not isinstance(result, list):
            return []

        positions = []
        for pos in result:
            size = float(pos.get('positionAmt', 0))
            if size != 0:  # Only include non-zero positions
                positions.append(Position(
                    symbol=pos['symbol'],
                    side='LONG' if size > 0 else 'SHORT',
                    size=abs(size),
                    entry_price=float(pos.get('entryPrice', 0)),
                    unrealized_pnl=float(pos.get('unRealizedProfit', 0)),
                    leverage=int(pos.get('leverage', 1)),
                    margin_type=pos.get('marginType', 'cross'),
                    liquidation_price=float(pos.get('liquidationPrice', 0))
                ))

        return positions

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        params = {
            'symbol': symbol,
            'leverage': leverage
        }
        result = self._make_request('POST', self.config.leverage_endpoint, params, signed=True)
        return 'error' not in result

    # ==================== Order Management ====================

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        position_side: PositionSide = PositionSide.BOTH,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None
    ) -> OrderResult:
        """
        Place a regular order

        For conditional orders (STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET),
        uses the new algo order endpoint as per Binance migration (2025-12-09)
        """
        # Validate and format quantity
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info:
            quantity = self._format_quantity(quantity, symbol_info)
            if price:
                price = self._format_price(price, symbol_info)
            if stop_price:
                stop_price = self._format_price(stop_price, symbol_info)

        # Check if this is a conditional order type (needs algo endpoint)
        conditional_types = [
            OrderType.STOP, OrderType.STOP_MARKET,
            OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_MARKET,
            OrderType.TRAILING_STOP_MARKET
        ]

        if order_type in conditional_types:
            return self._place_algo_conditional_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                stop_price=stop_price,
                price=price,
                position_side=position_side,
                reduce_only=reduce_only,
                client_order_id=client_order_id
            )

        # Standard order
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'quantity': quantity,
            'positionSide': position_side.value,
        }

        if order_type == OrderType.LIMIT:
            params['price'] = price
            params['timeInForce'] = time_in_force.value

        if reduce_only:
            params['reduceOnly'] = 'true'

        if client_order_id:
            params['newClientOrderId'] = client_order_id

        result = self._make_request('POST', self.config.order_endpoint, params, signed=True)

        if 'error' in result:
            return OrderResult(
                success=False,
                order_id=None,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                quantity=quantity,
                price=price,
                status='FAILED',
                message=result.get('msg', 'Unknown error'),
                raw_response=result
            )

        return OrderResult(
            success=True,
            order_id=str(result.get('orderId', '')),
            client_order_id=result.get('clientOrderId', client_order_id),
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            quantity=quantity,
            price=price,
            status=result.get('status', 'NEW'),
            message='Order placed successfully',
            raw_response=result
        )

    def _place_algo_conditional_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType,
        stop_price: float,
        price: Optional[float] = None,
        position_side: PositionSide = PositionSide.BOTH,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None
    ) -> OrderResult:
        """
        Place conditional order via the new algo order endpoint
        POST /fapi/v1/algoOrder

        This is required since 2025-12-09 migration
        """
        params = {
            'symbol': symbol,
            'side': side.value,
            'type': order_type.value,
            'quantity': quantity,
            'stopPrice': stop_price,
            'positionSide': position_side.value,
        }

        if price and order_type in [OrderType.STOP, OrderType.TAKE_PROFIT]:
            params['price'] = price

        if reduce_only:
            params['reduceOnly'] = 'true'

        if client_order_id:
            params['clientAlgoId'] = client_order_id

        result = self._make_request('POST', self.config.algo_order_endpoint, params, signed=True)

        if 'error' in result:
            return OrderResult(
                success=False,
                order_id=None,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                quantity=quantity,
                price=stop_price,
                status='FAILED',
                message=result.get('msg', 'Unknown error'),
                raw_response=result
            )

        return OrderResult(
            success=True,
            order_id=str(result.get('algoId', result.get('orderId', ''))),
            client_order_id=result.get('clientAlgoId', client_order_id),
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            quantity=quantity,
            price=stop_price,
            status=result.get('status', 'NEW'),
            message='Algo order placed successfully',
            raw_response=result
        )

    # ==================== TWAP Orders ====================

    def place_twap_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        duration_seconds: int = 600,
        position_side: PositionSide = PositionSide.BOTH,
        limit_price: Optional[float] = None,
        reduce_only: bool = False,
        client_algo_id: Optional[str] = None
    ) -> OrderResult:
        """
        Place a TWAP (Time-Weighted Average Price) order
        POST /sapi/v1/algo/futures/newOrderTwap

        TWAP splits large orders into smaller ones executed at regular intervals
        to minimize market impact.

        Requirements:
        - Duration: 5 min (300s) to 24 hours (86400s)
        - Notional: 1,000 USDT to 1,000,000 USDT
        - Max concurrent TWAP orders: 10
        """
        # Validate duration
        duration_seconds = max(
            self.algo_config.twap_min_duration,
            min(duration_seconds, self.algo_config.twap_max_duration)
        )

        # Format quantity
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info:
            quantity = self._format_quantity(quantity, symbol_info)
            if limit_price:
                limit_price = self._format_price(limit_price, symbol_info)

        params = {
            'symbol': symbol,
            'side': side.value,
            'quantity': quantity,
            'duration': duration_seconds,
            'positionSide': position_side.value,
        }

        if limit_price:
            params['limitPrice'] = limit_price

        if reduce_only:
            params['reduceOnly'] = 'true'

        if client_algo_id:
            params['clientAlgoId'] = client_algo_id
        else:
            # Generate a unique client algo ID
            params['clientAlgoId'] = f"twap_{symbol}_{self._get_timestamp()}"

        logger.info(f"Placing TWAP order: {symbol} {side.value} {quantity} over {duration_seconds}s")

        result = self._make_request(
            'POST',
            self.config.algo_twap_endpoint,
            params,
            signed=True,
            use_sapi=True
        )

        if 'error' in result or not result.get('success', False):
            return OrderResult(
                success=False,
                order_id=None,
                client_order_id=params['clientAlgoId'],
                symbol=symbol,
                side=side.value,
                order_type='TWAP',
                quantity=quantity,
                price=limit_price,
                status='FAILED',
                message=result.get('msg', 'TWAP order failed'),
                raw_response=result
            )

        return OrderResult(
            success=True,
            order_id=result.get('algoId', params['clientAlgoId']),
            client_order_id=result.get('clientAlgoId', params['clientAlgoId']),
            symbol=symbol,
            side=side.value,
            order_type='TWAP',
            quantity=quantity,
            price=limit_price,
            status='EXECUTING',
            message='TWAP order placed successfully',
            raw_response=result
        )

    def place_vp_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        urgency: str = "LOW",
        position_side: PositionSide = PositionSide.BOTH,
        limit_price: Optional[float] = None,
        reduce_only: bool = False,
        client_algo_id: Optional[str] = None
    ) -> OrderResult:
        """
        Place a VP (Volume Participation) order
        POST /sapi/v1/algo/futures/newOrderVp

        VP orders execute in proportion to real-time market volume.

        Requirements:
        - Notional: 1,000 USDT to 1,000,000 USDT
        - Urgency: LOW, MEDIUM, HIGH
        """
        # Format quantity
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info:
            quantity = self._format_quantity(quantity, symbol_info)
            if limit_price:
                limit_price = self._format_price(limit_price, symbol_info)

        params = {
            'symbol': symbol,
            'side': side.value,
            'quantity': quantity,
            'urgency': urgency.upper(),
            'positionSide': position_side.value,
        }

        if limit_price:
            params['limitPrice'] = limit_price

        if reduce_only:
            params['reduceOnly'] = 'true'

        if client_algo_id:
            params['clientAlgoId'] = client_algo_id
        else:
            params['clientAlgoId'] = f"vp_{symbol}_{self._get_timestamp()}"

        logger.info(f"Placing VP order: {symbol} {side.value} {quantity} urgency={urgency}")

        result = self._make_request(
            'POST',
            self.config.algo_vp_endpoint,
            params,
            signed=True,
            use_sapi=True
        )

        if 'error' in result or not result.get('success', False):
            return OrderResult(
                success=False,
                order_id=None,
                client_order_id=params['clientAlgoId'],
                symbol=symbol,
                side=side.value,
                order_type='VP',
                quantity=quantity,
                price=limit_price,
                status='FAILED',
                message=result.get('msg', 'VP order failed'),
                raw_response=result
            )

        return OrderResult(
            success=True,
            order_id=result.get('algoId', params['clientAlgoId']),
            client_order_id=result.get('clientAlgoId', params['clientAlgoId']),
            symbol=symbol,
            side=side.value,
            order_type='VP',
            quantity=quantity,
            price=limit_price,
            status='EXECUTING',
            message='VP order placed successfully',
            raw_response=result
        )

    # ==================== Algo Order Queries ====================

    def get_algo_order(self, algo_id: Optional[str] = None, client_algo_id: Optional[str] = None) -> Dict:
        """Query a specific algo order"""
        params = {}
        if algo_id:
            params['algoId'] = algo_id
        elif client_algo_id:
            params['clientAlgoId'] = client_algo_id
        else:
            return {'error': True, 'msg': 'Either algoId or clientAlgoId required'}

        return self._make_request('GET', self.config.algo_query_endpoint, params, signed=True)

    def get_open_algo_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open algo orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol

        result = self._make_request('GET', self.config.algo_open_orders_endpoint, params, signed=True)

        if 'error' in result:
            return []

        return result.get('orders', []) if isinstance(result, dict) else result

    def cancel_algo_order(self, algo_id: str) -> bool:
        """Cancel an algo order"""
        params = {'algoId': algo_id}
        result = self._make_request('DELETE', '/fapi/v1/algoOrder', params, signed=True)
        return 'error' not in result

    # ==================== Smart Order Execution ====================

    def execute_smart_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        position_side: PositionSide = PositionSide.BOTH,
        use_twap: Optional[bool] = None,
        twap_duration: Optional[int] = None
    ) -> OrderResult:
        """
        Execute an order with smart routing

        Automatically decides between:
        - Direct market order for small orders
        - TWAP for large orders to minimize slippage
        """
        # Get current price to calculate notional
        current_price = self.get_ticker_price(symbol)
        if not current_price:
            return OrderResult(
                success=False,
                order_id=None,
                client_order_id=None,
                symbol=symbol,
                side=side.value,
                order_type='SMART',
                quantity=quantity,
                price=None,
                status='FAILED',
                message='Failed to get current price',
                raw_response={}
            )

        notional_value = quantity * current_price

        # Decide order type
        if use_twap is None:
            use_twap = (
                self.algo_config.use_twap_for_large_orders and
                notional_value >= self.algo_config.twap_threshold_usdt
            )

        if use_twap and notional_value >= 1000:  # TWAP minimum is 1000 USDT
            duration = twap_duration or self.algo_config.twap_duration_seconds
            logger.info(f"Using TWAP for {symbol} order (notional: ${notional_value:.2f})")
            return self.place_twap_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                duration_seconds=duration,
                position_side=position_side
            )
        else:
            logger.info(f"Using market order for {symbol} (notional: ${notional_value:.2f})")
            return self.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                position_side=position_side
            )

    # ==================== Utility Methods ====================

    def _format_quantity(self, quantity: float, symbol_info: SymbolInfo) -> float:
        """Format quantity according to symbol rules"""
        step_size = symbol_info.step_size
        precision = symbol_info.quantity_precision
        quantity = round(quantity - (quantity % step_size), precision)
        return max(quantity, symbol_info.min_qty)

    def _format_price(self, price: float, symbol_info: SymbolInfo) -> float:
        """Format price according to symbol rules"""
        tick_size = symbol_info.tick_size
        precision = symbol_info.price_precision
        return round(price - (price % tick_size), precision)

    def validate_order(
        self,
        symbol: str,
        quantity: float,
        price: Optional[float] = None
    ) -> tuple[bool, str]:
        """Validate order parameters before submission"""
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return False, f"Symbol {symbol} not found"

        if quantity < symbol_info.min_qty:
            return False, f"Quantity {quantity} below minimum {symbol_info.min_qty}"

        if quantity > symbol_info.max_qty:
            return False, f"Quantity {quantity} above maximum {symbol_info.max_qty}"

        current_price = price or self.get_ticker_price(symbol)
        if current_price:
            notional = quantity * current_price
            if notional < symbol_info.min_notional:
                return False, f"Notional {notional} below minimum {symbol_info.min_notional}"

        return True, "Valid"

    def close_position(
        self,
        symbol: str,
        position_side: PositionSide = PositionSide.BOTH
    ) -> OrderResult:
        """Close an existing position"""
        positions = self.get_positions(symbol)

        for pos in positions:
            if position_side == PositionSide.BOTH or pos.side == position_side.value:
                # Determine close side
                close_side = OrderSide.SELL if pos.side == 'LONG' else OrderSide.BUY

                return self.place_order(
                    symbol=symbol,
                    side=close_side,
                    quantity=pos.size,
                    order_type=OrderType.MARKET,
                    position_side=position_side,
                    reduce_only=True
                )

        return OrderResult(
            success=False,
            order_id=None,
            client_order_id=None,
            symbol=symbol,
            side='',
            order_type='MARKET',
            quantity=0,
            price=None,
            status='NO_POSITION',
            message='No position found to close',
            raw_response={}
        )
