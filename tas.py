import requests
import json
import time
import logging
import threading
import queue
import traceback
from datetime import datetime, timedelta
import matplotlib

# Use non-GUI backend for thread safety
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import numpy as np
from collections import deque
import os
import sys

# Import API keys
try:
    from apis import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    print("ERROR: Critical - apis.py missing or TELEGRAM_API_TOKEN/TELEGRAM_CHAT_ID not found.")
    sys.exit(1) # Exit if Telegram keys are not found

try:
    from apis import BYBIT_API_KEY, BYBIT_SECRET_KEY
    print("INFO: Bybit API keys found.")
except ImportError:
    BYBIT_API_KEY, BYBIT_SECRET_KEY = None, None
    print("INFO: Bybit API keys not found in apis.py.")

try:
    from apis import BINGX_API_KEY, BINGX_SECRET_KEY
    print("INFO: BingX API keys found.")
except ImportError:
    BINGX_API_KEY, BINGX_SECRET_KEY = None, None
    print("INFO: BingX API keys not found in apis.py.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter to prevent API throttling"""
    
    def __init__(self, max_requests=10, time_window=1):
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
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                
                # Clean up after sleeping
                now = time.time()
                while self.requests and self.requests[0] < now - self.time_window:
                    self.requests.popleft()
            
            self.requests.append(now)

class BybitClient:
    """Client for Bybit API interactions"""
    
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.rate_limiter = RateLimiter(max_requests=10, time_window=1)
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CryptoTradingBot/1.0'
        })

    def make_request(self, endpoint, params=None, max_retries=5):
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                logger.debug(f"Making request to {endpoint} (attempt {attempt + 1}/{max_retries})")
                
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ret_code') == 0 or data.get('retCode') == 0:
                        return data.get('result') or data.get('data')
                    else:
                        logger.warning(f"API error: {data.get('ret_msg') or data.get('retMsg')}")
                elif response.status_code == 429:
                    logger.warning("Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                else:
                    logger.warning(f"HTTP error {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
            
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to get data from {endpoint} after {max_retries} attempts")
        return None

    def get_symbols(self):
        """Get all USDT perpetual futures symbols"""
        logger.info("Fetching all USDT futures symbols from Bybit...")
        
        data = self.make_request("/v5/market/instruments-info", params={"category": "linear"})
        if not data:
            return []
        
        symbols = []
        for item in data.get('list', []):
            if item.get('settleCoin') == 'USDT' and item.get('status') == 'Trading':
                symbols.append(item['symbol'])
        
        logger.info(f"Found {len(symbols)} active USDT futures symbols")
        return symbols

    def get_klines(self, symbol, interval='15', limit=200):
        """Get candlestick data"""
        logger.debug(f"Fetching klines for {symbol}")
        
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        data = self.make_request("/v5/market/kline", params=params)
        if not data or 'list' not in data:
            return None
        
        # Convert to structured format
        klines = []
        for candle in reversed(data['list']):  # Reverse to get chronological order
            klines.append({
                'timestamp': int(candle[0]),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            })
        
        return klines

class BingXClient:
    """Client for BingX API interactions"""

    def __init__(self, api_key=None, secret_key=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://open-api.bingx.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CryptoTradingBot/1.0'
        })
        if self.api_key:
            self.session.headers.update({'X-BX-APIKEY': self.api_key})

        # Using a slightly more conservative rate limit for BingX as an initial guess
        self.rate_limiter = RateLimiter(max_requests=5, time_window=1)

    def _sign_request(self, params):
        """Placeholder for signing requests. Not used for public GET endpoints if API key in header is sufficient."""
        # In a real scenario, this would involve:
        # 1. Adding a timestamp to params.
        # 2. Sorting params alphabetically.
        # 3. Creating a query string.
        # 4. Signing the string with HMAC-SHA256 using self.secret_key.
        # 5. Adding the signature to params or headers.
        # For now, as get_symbols and get_klines are often public, we assume API key in header is enough.
        # If secret_key is present, one might add a signature parameter, e.g.
        # if self.secret_key:
        #     # Simplified: actual signature generation would be more complex
        #     # query_string = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
        #     # signature = hmac.new(self.secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        #     # params['signature'] = signature
        #     pass # No actual signing implemented for this placeholder
        return params

    def make_request(self, method, endpoint, params=None, max_retries=5):
        """Make HTTP request with retry logic for BingX"""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        # Apply signing if needed (though current public methods might not need it)
        # params = self._sign_request(params)

        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                logger.debug(f"Making BingX request to {endpoint} with params {params} (attempt {attempt + 1}/{max_retries})")

                response = self.session.request(method.upper(), url, params=params if method.upper() == 'GET' else None,
                                                json=params if method.upper() != 'GET' else None, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    # BingX success code is typically 0
                    if data.get('code') == 0:
                        return data.get('data') # Common for BingX
                    else:
                        logger.warning(f"BingX API error: Code {data.get('code')}, Msg: {data.get('msg')}")
                elif response.status_code == 429: # Too Many Requests
                    logger.warning("BingX rate limit hit (429), waiting 60 seconds...")
                    time.sleep(60) # Standard wait time for rate limit
                elif response.status_code == 401: # Unauthorized
                     logger.error(f"BingX API authentication error (401): {response.text}. Check API key and permissions.")
                     break # No point retrying auth errors usually
                elif response.status_code == 403: # Forbidden
                     logger.error(f"BingX API access forbidden (403): {response.text}. Check IP whitelist or endpoint permissions.")
                     break # No point retrying auth errors usually
                else:
                    logger.warning(f"BingX HTTP error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                logger.error(f"BingX request failed: {e}")
            except Exception as e: # Catch any other unexpected errors
                logger.error(f"Unexpected error during BingX request: {e}")

            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30) # Exponential backoff
                logger.info(f"Retrying BingX request in {wait_time} seconds...")
                time.sleep(wait_time)

        logger.error(f"Failed to get data from BingX {endpoint} after {max_retries} attempts")
        return None

    def get_symbols(self):
        """Get all USDT perpetual futures symbols from BingX"""
        logger.info("Fetching all USDT perpetual futures symbols from BingX...")
        # Endpoint based on common BingX API patterns for perpetual swap contracts
        endpoint = "/openApi/swap/v2/quote/contracts"

        data = self.make_request("GET", endpoint)

        if not data:
            logger.error("No data received from BingX get_symbols.")
            return []

        symbols = []
        # Assuming data is a list of contract details directly if 'data' key holds the list
        # Or it could be data.get('list') or similar, adjust based on actual response
        contract_list = data if isinstance(data, list) else data.get('list', [])

        for item in contract_list:
            # Assuming fields: 'symbol', 'status', 'quoteCoin' or 'marginAsset'
            # These field names are guesses and might need adjustment
            if (item.get('quoteCoin') == 'USDT' or item.get('marginAsset') == 'USDT') and \
               item.get('status') == 'Trading':
                symbols.append(item['symbol']) # e.g., "BTC-USDT"

        logger.info(f"Found {len(symbols)} active USDT futures symbols on BingX")
        return symbols

    def get_klines(self, symbol, interval='15m', limit=200):
        """Get candlestick data from BingX"""
        logger.debug(f"Fetching klines for {symbol} from BingX")
        # Endpoint based on common BingX API patterns
        endpoint = "/openApi/swap/v2/quote/klines"

        # Map common interval to BingX format if necessary.
        # For now, assume '15m' is accepted. BingX might use 'Min15', 'H1', etc.
        # Example mapping: interval_map = {'15': 'Min15', '1h': 'H1'}
        params = {
            'symbol': symbol,
            'interval': interval, # e.g., "15m", "1h", "4h", "1d"
            'limit': limit
        }

        data = self.make_request("GET", endpoint, params=params)

        if not data:
            logger.warning(f"No kline data for {symbol} from BingX.")
            return None

        # Assuming data is a list of lists: [[ts, open, high, low, close, volume, ...], ...]
        # And BingX returns newest first, so we might need to reverse.
        # Or it might be oldest first. This needs to be checked with actual API response.
        # For consistency with Bybit client, let's assume we want chronological (oldest first)

        klines = []
        # The actual data might be nested, e.g. data.get('list')
        kline_list = data if isinstance(data, list) else data.get('list', [])

        # If BingX returns newest first, reverse it:
        # if kline_list and len(kline_list) > 1 and int(kline_list[0][0]) > int(kline_list[-1][0]):
        #    kline_list.reverse()

        for candle in kline_list:
            if len(candle) >= 6: # Ensure we have enough data points
                try:
                    klines.append({
                        'timestamp': int(candle[0]),       # Timestamp (ms)
                        'open': float(candle[1]),         # Open
                        'high': float(candle[2]),         # High
                        'low': float(candle[3]),          # Low
                        'close': float(candle[4]),        # Close
                        'volume': float(candle[5])        # Volume
                        # BingX might have 'turnover' or 'quoteAssetVolume' as candle[6] or candle[7]
                    })
                except (ValueError, TypeError) as e:
                    logger.error(f"Error parsing kline candle for {symbol}: {candle} - {e}")
            else:
                logger.warning(f"Skipping malformed kline candle for {symbol}: {candle}")

        # If data was newest first, and we didn't reverse above, do it now.
        # Or, if data is oldest first, this is fine.
        # Let's assume for now it's returned oldest first or the make_request sorts it.
        # The Bybit client reverses if it gets newest first. We should aim for consistency.
        # If BingX returns newest first, uncomment the reverse() above or do it here.
        # Example: if klines and len(klines) > 1 and klines[0]['timestamp'] > klines[-1]['timestamp']:
        #    klines.reverse()

        return klines

class TechnicalIndicators:
    """Calculate technical indicators without using talib"""
    
    @staticmethod
    def sma(data, period):
        """Simple Moving Average"""
        if len(data) < period:
            return [None] * len(data)
        
        sma_values = []
        for i in range(len(data)):
            if i < period - 1:
                sma_values.append(None)
            else:
                sma_values.append(sum(data[i - period + 1:i + 1]) / period)
        
        return sma_values

    @staticmethod
    def ema(data, period):
        """Exponential Moving Average"""
        if len(data) < period:
            return [None] * len(data)
        
        multiplier = 2 / (period + 1)
        ema_values = [None] * (period - 1)
        
        # First EMA is SMA
        ema_values.append(sum(data[:period]) / period)
        
        # Calculate rest
        for i in range(period, len(data)):
            ema_values.append((data[i] - ema_values[-1]) * multiplier + ema_values[-1])
        
        return ema_values

    @staticmethod
    def rsi(data, period=14):
        """Relative Strength Index"""
        if len(data) < period + 1:
            return [None] * len(data)
        
        gains = []
        losses = []
        
        for i in range(1, len(data)):
            diff = data[i] - data[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))
        
        rsi_values = [None] * (period + 1)
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values

    @staticmethod
    def stochastic(highs, lows, closes, k_period=14, d_period=3):
        """Stochastic oscillator"""
        k_values = []
        
        for i in range(len(closes)):
            if i < k_period - 1:
                k_values.append(None)
            else:
                highest = max(highs[i - k_period + 1:i + 1])
                lowest = min(lows[i - k_period + 1:i + 1])
                
                if highest == lowest:
                    k_values.append(50)
                else:
                    k = ((closes[i] - lowest) / (highest - lowest)) * 100
                    k_values.append(k)
        
        # Calculate %D (SMA of %K)
        d_values = TechnicalIndicators.sma([k for k in k_values if k is not None], d_period)
        
        # Reconstruct with None values
        d_final = []
        j = 0
        for i in range(len(k_values)):
            if k_values[i] is None:
                d_final.append(None)
            else:
                d_final.append(d_values[j] if j < len(d_values) else None)
                j += 1
        
        return k_values, d_final

    @staticmethod
    def atr(highs, lows, closes, period=14):
        """Average True Range"""
        true_ranges = []
        
        for i in range(len(closes)):
            if i == 0:
                true_ranges.append(highs[i] - lows[i])
            else:
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                true_ranges.append(tr)
        
        # Calculate ATR using EMA
        return TechnicalIndicators.ema(true_ranges, period)

    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Bollinger Bands"""
        sma_values = TechnicalIndicators.sma(data, period)
        upper_band = []
        lower_band = []
        
        for i in range(len(data)):
            if sma_values[i] is None:
                upper_band.append(None)
                lower_band.append(None)
            else:
                std = np.std(data[max(0, i - period + 1):i + 1])
                upper_band.append(sma_values[i] + std_dev * std)
                lower_band.append(sma_values[i] - std_dev * std)
        
        return upper_band, sma_values, lower_band

    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = []
        for i in range(len(data)):
            if ema_fast[i] is None or ema_slow[i] is None:
                macd_line.append(None)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        # Filter out None values for signal calculation
        macd_filtered = [x for x in macd_line if x is not None]
        signal_line_filtered = TechnicalIndicators.ema(macd_filtered, signal)
        
        # Reconstruct signal line with None values
        signal_line = []
        j = 0
        for i in range(len(macd_line)):
            if macd_line[i] is None:
                signal_line.append(None)
            else:
                signal_line.append(signal_line_filtered[j] if j < len(signal_line_filtered) else None)
                j += 1
        
        histogram = []
        for i in range(len(macd_line)):
            if macd_line[i] is None or signal_line[i] is None:
                histogram.append(None)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return macd_line, signal_line, histogram

class CandlePatterns:
    """Identify candlestick patterns for fine-tuned entries"""
    
    @staticmethod
    def is_doji(open_price, high, low, close):
        """Identify doji candle"""
        body = abs(close - open_price)
        range_hl = high - low
        
        if range_hl == 0:
            return False
        
        return body / range_hl < 0.1

    @staticmethod
    def is_hammer(open_price, high, low, close):
        """Identify hammer pattern"""
        body = abs(close - open_price)
        range_hl = high - low
        
        if range_hl == 0:
            return False
        
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        return (lower_shadow > body * 2 and
                upper_shadow < body * 0.5 and
                body / range_hl > 0.1)

    @staticmethod
    def is_shooting_star(open_price, high, low, close):
        """Identify shooting star pattern"""
        body = abs(close - open_price)
        range_hl = high - low
        
        if range_hl == 0:
            return False
        
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        
        return (upper_shadow > body * 2 and
                lower_shadow < body * 0.5 and
                body / range_hl > 0.1)

    @staticmethod
    def is_engulfing_bullish(prev_open, prev_close, curr_open, curr_close):
        """Identify bullish engulfing pattern"""
        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open
        
        return (prev_bearish and curr_bullish and
                curr_open <= prev_close and
                curr_close >= prev_open)

    @staticmethod
    def is_engulfing_bearish(prev_open, prev_close, curr_open, curr_close):
        """Identify bearish engulfing pattern"""
        prev_bullish = prev_close > prev_open
        curr_bearish = curr_close < curr_open
        
        return (prev_bullish and curr_bearish and
                curr_open >= prev_close and
                curr_close <= prev_open)

class TradingStrategy:
    """Advanced trading strategy with fine-tuned entries"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.patterns = CandlePatterns()

    def calculate_pivot_points(self, high, low, close):
        """Calculate pivot points for support/resistance"""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }

    def find_support_resistance(self, highs, lows, window=20, min_touches=2):
        """Find dynamic support and resistance levels"""
        levels = []
        
        # Find swing highs and lows
        for i in range(window, len(highs) - window):
            # Swing high
            if highs[i] == max(highs[i-window:i+window+1]):
                level = highs[i]
                touches = sum(1 for h in highs[i+1:] if abs(h - level) / level < 0.001)
                if touches >= min_touches:
                    levels.append(('resistance', level, touches))
            
            # Swing low
            if lows[i] == min(lows[i-window:i+window+1]):
                level = lows[i]
                touches = sum(1 for l in lows[i+1:] if abs(l - level) / level < 0.001)
                if touches >= min_touches:
                    levels.append(('support', level, touches))
        
        return levels

    def calculate_momentum_score(self, closes, volumes, rsi, macd_hist):
        """Calculate momentum score for entry timing"""
        score = 0
        
        # Price momentum
        if len(closes) >= 5:
            recent_momentum = (closes[-1] - closes[-5]) / closes[-5]
            if abs(recent_momentum) > 0.005:  # 0.5% move
                score += 1 if recent_momentum > 0 else -1
        
        # Volume momentum
        if len(volumes) >= 20:
            vol_avg = np.mean(volumes[-20:-1])
            if volumes[-1] > vol_avg * 1.5:
                score += 1
            elif volumes[-1] < vol_avg * 0.5:
                score -= 1
        
        # RSI momentum
        if rsi[-1] is not None and len(rsi) >= 3:
            if all(rsi[-3:]):
                rsi_momentum = rsi[-1] - rsi[-3]
                if rsi_momentum > 5:
                    score += 1
                elif rsi_momentum < -5:
                    score -= 1
        
        # MACD momentum
        if len(macd_hist) >= 3 and all(macd_hist[-3:]):
            if macd_hist[-1] > macd_hist[-2] > macd_hist[-3]:
                score += 1
            elif macd_hist[-1] < macd_hist[-2] < macd_hist[-3]:
                score -= 1
        
        return score

    def find_optimal_entry(self, signal_type, current_price, atr_value, support_levels, resistance_levels):
        """Find optimal entry point based on market structure"""
        entry_adjustments = []
        
        if signal_type == "BUY":
            # Look for nearby support for better entry
            nearby_supports = [s[1] for s in support_levels
                             if s[0] == 'support' and
                             current_price * 0.99 <= s[1] <= current_price * 1.005]
            
            if nearby_supports:
                # Enter at support level
                optimal_support = max(nearby_supports)
                entry_adjustments.append({
                    'type': 'support_entry',
                    'price': optimal_support,
                    'reason': 'Entry at support level'
                })
            
            # Fibonacci retracement entry
            if len(resistance_levels) > 0:
                nearest_resistance = min([r[1] for r in resistance_levels
                                        if r[0] == 'resistance' and r[1] > current_price],
                                       default=current_price * 1.05)
                
                fib_382 = current_price + (nearest_resistance - current_price) * 0.382
                fib_618 = current_price + (nearest_resistance - current_price) * 0.618
                
                entry_adjustments.append({
                    'type': 'fibonacci_entry',
                    'price': current_price,
                    'target1': fib_382,
                    'target2': fib_618,
                    'reason': 'Fibonacci extension targets'
                })
            
            # ATR-based pullback entry
            if atr_value:
                pullback_entry = current_price - (atr_value * 0.5)
                entry_adjustments.append({
                    'type': 'atr_pullback',
                    'price': pullback_entry,
                    'reason': '50% ATR pullback entry'
                })
        
        else:  # SELL signal
            # Look for nearby resistance for better entry
            nearby_resistances = [r[1] for r in resistance_levels
                                if r[0] == 'resistance' and
                                current_price * 0.995 <= r[1] <= current_price * 1.01]
            
            if nearby_resistances:
                # Enter at resistance level
                optimal_resistance = min(nearby_resistances)
                entry_adjustments.append({
                    'type': 'resistance_entry',
                    'price': optimal_resistance,
                    'reason': 'Entry at resistance level'
                })
            
            # ATR-based pullback entry
            if atr_value:
                pullback_entry = current_price + (atr_value * 0.5)
                entry_adjustments.append({
                    'type': 'atr_pullback',
                    'price': pullback_entry,
                    'reason': '50% ATR pullback entry'
                })
        
        return entry_adjustments

    def analyze(self, klines):
        """Analyze price data and generate trading signals with fine-tuned entries"""
        if not klines or len(klines) < 100:
            return None
        
        # Extract price data
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        opens = [k['open'] for k in klines]
        volumes = [k['volume'] for k in klines]
        
        # Calculate indicators
        logger.debug("Calculating technical indicators...")
        
        # Moving averages
        sma_20 = self.indicators.sma(closes, 20)
        sma_50 = self.indicators.sma(closes, 50)
        ema_9 = self.indicators.ema(closes, 9)
        ema_21 = self.indicators.ema(closes, 21)
        
        # RSI
        rsi = self.indicators.rsi(closes)
        
        # Stochastic
        stoch_k, stoch_d = self.indicators.stochastic(highs, lows, closes)
        
        # ATR for volatility-based stops
        atr = self.indicators.atr(highs, lows, closes)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(closes)
        
        # MACD
        macd_line, signal_line, histogram = self.indicators.macd(closes)
        
        # Volume analysis
        volume_sma = self.indicators.sma(volumes, 20)
        
        # Find support and resistance levels
        support_resistance = self.find_support_resistance(highs, lows)
        
        # Calculate pivot points
        pivot_points = self.calculate_pivot_points(highs[-1], lows[-1], closes[-1])
        
        # Get latest values
        current_price = closes[-1]
        current_rsi = rsi[-1] if rsi[-1] is not None else 50
        current_stoch_k = stoch_k[-1] if stoch_k[-1] is not None else 50
        current_stoch_d = stoch_d[-1] if stoch_d[-1] is not None else 50
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1] if volume_sma[-1] is not None else current_volume
        current_atr = atr[-1] if atr[-1] is not None else None
        
        # Check candlestick patterns
        current_candle = {
            'doji': self.patterns.is_doji(opens[-1], highs[-1], lows[-1], closes[-1]),
            'hammer': self.patterns.is_hammer(opens[-1], highs[-1], lows[-1], closes[-1]),
            'shooting_star': self.patterns.is_shooting_star(opens[-1], highs[-1], lows[-1], closes[-1])
        }
        
        if len(opens) >= 2:
            current_candle['bullish_engulfing'] = self.patterns.is_engulfing_bullish(
                opens[-2], closes[-2], opens[-1], closes[-1])
            current_candle['bearish_engulfing'] = self.patterns.is_engulfing_bearish(
                opens[-2], closes[-2], opens[-1], closes[-1])
        
        # Calculate momentum score
        momentum_score = self.calculate_momentum_score(closes, volumes, rsi, histogram)
        
        # Initialize signal
        signal = None
        reasons = []
        strength = 0
        entry_conditions = []
        signal_explanation = []
        timeframe_info = "15m"  # Current timeframe
        
        # Strategy 1: RSI Divergence with Stochastic confirmation
        if current_rsi < 30 and current_stoch_k < 20 and current_stoch_k > current_stoch_d:
            signal = "BUY"
            reasons.append(f"RSI oversold ({current_rsi:.2f}) with Stochastic bullish cross")
            strength += 2
            entry_conditions.append("Wait for price to close above EMA 9")
            signal_explanation.append(f"✅ RSI below 30 ({current_rsi:.1f}) - Market oversold")
            signal_explanation.append(f"✅ Stochastic %K below 20 ({current_stoch_k:.1f}) - Extreme oversold")
            signal_explanation.append(f"✅ Stochastic bullish cross (%K > %D) - Momentum turning bullish")
            
            # Fine-tune: Look for hammer or bullish engulfing
            if current_candle.get('hammer') or current_candle.get('bullish_engulfing'):
                strength += 1
                entry_conditions.append("Bullish candlestick pattern confirmed")
                if current_candle.get('hammer'):
                    signal_explanation.append("✅ Hammer candlestick pattern - Rejection at lows")
                if current_candle.get('bullish_engulfing'):
                    signal_explanation.append("✅ Bullish engulfing pattern - Strong buying pressure")
        
        elif current_rsi > 70 and current_stoch_k > 80 and current_stoch_k < current_stoch_d:
            signal = "SELL"
            reasons.append(f"RSI overbought ({current_rsi:.2f}) with Stochastic bearish cross")
            strength += 2
            entry_conditions.append("Wait for price to close below EMA 9")
            signal_explanation.append(f"✅ RSI above 70 ({current_rsi:.1f}) - Market overbought")
            signal_explanation.append(f"✅ Stochastic %K above 80 ({current_stoch_k:.1f}) - Extreme overbought")
            signal_explanation.append(f"✅ Stochastic bearish cross (%K < %D) - Momentum turning bearish")
            
            # Fine-tune: Look for shooting star or bearish engulfing
            if current_candle.get('shooting_star') or current_candle.get('bearish_engulfing'):
                strength += 1
                entry_conditions.append("Bearish candlestick pattern confirmed")
                if current_candle.get('shooting_star'):
                    signal_explanation.append("✅ Shooting star pattern - Rejection at highs")
                if current_candle.get('bearish_engulfing'):
                    signal_explanation.append("✅ Bearish engulfing pattern - Strong selling pressure")
        
        # Strategy 2: Bollinger Band squeeze with momentum
        if bb_upper[-1] and bb_lower[-1] and current_atr:
            bb_width = bb_upper[-1] - bb_lower[-1]
            bb_width_avg = np.mean([bb_upper[i] - bb_lower[i] for i in range(-20, 0)
                                  if bb_upper[i] and bb_lower[i]])
            
            if bb_width < bb_width_avg * 0.7:  # Tighter squeeze
                if current_price > bb_middle[-1] and momentum_score > 0:
                    if signal != "SELL":
                        signal = "BUY"
                        reasons.append("BB squeeze with bullish momentum")
                        strength += 2
                        entry_conditions.append("Enter on breakout above BB upper band")
                        entry_conditions.append(f"Momentum score: {momentum_score}")
                        signal_explanation.append(f"✅ Bollinger Band squeeze detected - Low volatility ({bb_width/bb_width_avg:.1%} of average)")
                        signal_explanation.append(f"✅ Price above BB middle line - Bullish bias")
                        signal_explanation.append(f"✅ Positive momentum score ({momentum_score}) - Multiple confirmations")
                
                elif current_price < bb_middle[-1] and momentum_score < 0:
                    if signal != "BUY":
                        signal = "SELL"
                        reasons.append("BB squeeze with bearish momentum")
                        strength += 2
                        entry_conditions.append("Enter on breakdown below BB lower band")
                        entry_conditions.append(f"Momentum score: {momentum_score}")
                        signal_explanation.append(f"✅ Bollinger Band squeeze detected - Low volatility ({bb_width/bb_width_avg:.1%} of average)")
                        signal_explanation.append(f"✅ Price below BB middle line - Bearish bias")
                        signal_explanation.append(f"✅ Negative momentum score ({momentum_score}) - Multiple confirmations")
        
        # Strategy 3: MACD with volume surge
        if len(macd_line) >= 3 and all(macd_line[-3:]) and all(signal_line[-3:]):
            macd_cross_up = macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]
            macd_cross_down = macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]
            
            if macd_cross_up and current_volume > avg_volume * 2:
                if signal != "SELL":
                    signal = "BUY"
                    reasons.append("MACD bullish cross with volume surge")
                    strength += 2
                    entry_conditions.append("Enter after MACD histogram turns positive")
                    signal_explanation.append(f"✅ MACD bullish crossover - MACD ({macd_line[-1]:.6f}) > Signal ({signal_line[-1]:.6f})")
                    signal_explanation.append(f"✅ Volume surge - {current_volume/avg_volume:.1f}x average volume")
                    signal_explanation.append("✅ Momentum shifting from bearish to bullish")
            
            elif macd_cross_down and current_volume > avg_volume * 2:
                if signal != "BUY":
                    signal = "SELL"
                    reasons.append("MACD bearish cross with volume surge")
                    strength += 2
                    entry_conditions.append("Enter after MACD histogram turns negative")
                    signal_explanation.append(f"✅ MACD bearish crossover - MACD ({macd_line[-1]:.6f}) < Signal ({signal_line[-1]:.6f})")
                    signal_explanation.append(f"✅ Volume surge - {current_volume/avg_volume:.1f}x average volume")
                    signal_explanation.append("✅ Momentum shifting from bullish to bearish")
        
        # Strategy 4: Support/Resistance breakout with retest
        support_levels = [s for s in support_resistance if s[0] == 'support']
        resistance_levels = [s for s in support_resistance if s[0] == 'resistance']
        
        if resistance_levels:
            nearest_resistance = min([r[1] for r in resistance_levels if r[1] > current_price * 0.999],
                                   default=None)
            
            if nearest_resistance and current_price > nearest_resistance * 0.998:
                if signal != "SELL":
                    signal = "BUY"
                    reasons.append(f"Breaking resistance at {nearest_resistance:.4f}")
                    strength += 1
                    entry_conditions.append("Wait for retest of broken resistance as support")
                    entry_conditions.append(f"Entry zone: {nearest_resistance * 0.998:.4f} - {nearest_resistance * 1.002:.4f}")
                    signal_explanation.append(f"✅ Breaking key resistance level at ${nearest_resistance:.4f}")
                    signal_explanation.append("✅ Resistance becomes support - Role reversal")
                    signal_explanation.append("✅ Breakout indicates bullish momentum")
        
        if support_levels:
            nearest_support = max([s[1] for s in support_levels if s[1] < current_price * 1.001],
                                 default=None)
            
            if nearest_support and current_price < nearest_support * 1.002:
                if signal != "BUY":
                    signal = "SELL"
                    reasons.append(f"Breaking support at {nearest_support:.4f}")
                    strength += 1
                    entry_conditions.append("Wait for retest of broken support as resistance")
                    entry_conditions.append(f"Entry zone: {nearest_support * 0.998:.4f} - {nearest_support * 1.002:.4f}")
                    signal_explanation.append(f"✅ Breaking key support level at ${nearest_support:.4f}")
                    signal_explanation.append("✅ Support becomes resistance - Role reversal")
                    signal_explanation.append("✅ Breakdown indicates bearish momentum")
        
        # Strategy 5: EMA crossover with trend alignment
        if all(ema_9[-3:]) and all(ema_21[-3:]) and all(sma_50[-3:]):
            ema_cross_up = ema_9[-1] > ema_21[-1] and ema_9[-2] <= ema_21[-2]
            ema_cross_down = ema_9[-1] < ema_21[-1] and ema_9[-2] >= ema_21[-2]
            
            if ema_cross_up and current_price > sma_50[-1]:
                if signal != "SELL":
                    signal = "BUY"
                    reasons.append("EMA 9/21 bullish cross above SMA 50")
                    strength += 2
                    entry_conditions.append("Enter on pullback to EMA 9")
                    if current_atr:
                        entry_conditions.append(f"Ideal entry: {current_price - current_atr * 0.3:.4f}")
                    signal_explanation.append(f"✅ EMA 9 ({ema_9[-1]:.4f}) crossed above EMA 21 ({ema_21[-1]:.4f})")
                    signal_explanation.append(f"✅ Price above SMA 50 ({sma_50[-1]:.4f}) - Confirming uptrend")
                    signal_explanation.append("✅ Short-term momentum aligns with longer-term trend")
            
            elif ema_cross_down and current_price < sma_50[-1]:
                if signal != "BUY":
                    signal = "SELL"
                    reasons.append("EMA 9/21 bearish cross below SMA 50")
                    strength += 2
                    entry_conditions.append("Enter on pullback to EMA 9")
                    if current_atr:
                        entry_conditions.append(f"Ideal entry: {current_price + current_atr * 0.3:.4f}")
                    signal_explanation.append(f"✅ EMA 9 ({ema_9[-1]:.4f}) crossed below EMA 21 ({ema_21[-1]:.4f})")
                    signal_explanation.append(f"✅ Price below SMA 50 ({sma_50[-1]:.4f}) - Confirming downtrend")
                    signal_explanation.append("✅ Short-term momentum aligns with longer-term trend")
        
        # Only return signal if strength is sufficient
        if signal and strength >= 3:
            # Find optimal entries
            entry_adjustments = self.find_optimal_entry(
                signal, current_price, current_atr, support_levels, resistance_levels
            )
            
            # Calculate dynamic stop loss and take profit based on ATR
            if signal == "BUY":
                if current_atr:
                    stop_loss = current_price - (current_atr * 1.5)
                    take_profit1 = current_price + (current_atr * 2)
                    take_profit2 = current_price + (current_atr * 3)
                    take_profit3 = current_price + (current_atr * 5)
                else:
                    stop_loss = current_price * 0.98
                    take_profit1 = current_price * 1.02
                    take_profit2 = current_price * 1.03
                    take_profit3 = current_price * 1.05
                
                # Adjust stop loss to below nearest support if available
                if support_levels:
                    nearest_support = max([s[1] for s in support_levels if s[1] < current_price],
                                         default=None)
                    if nearest_support:
                        stop_loss = min(stop_loss, nearest_support * 0.997)
            
            else:  # SELL
                if current_atr:
                    stop_loss = current_price + (current_atr * 1.5)
                    take_profit1 = current_price - (current_atr * 2)
                    take_profit2 = current_price - (current_atr * 3)
                    take_profit3 = current_price - (current_atr * 5)
                else:
                    stop_loss = current_price * 1.02
                    take_profit1 = current_price * 0.98
                    take_profit2 = current_price * 0.97
                    take_profit3 = current_price * 0.95
                
                # Adjust stop loss to above nearest resistance if available
                if resistance_levels:
                    nearest_resistance = min([r[1] for r in resistance_levels if r[1] > current_price],
                                            default=None)
                    if nearest_resistance:
                        stop_loss = max(stop_loss, nearest_resistance * 1.003)
            
            return {
                'signal': signal,
                'reasons': reasons,
                'strength': strength,
                'entry': current_price,
                'stop_loss': stop_loss,
                'take_profit1': take_profit1,
                'take_profit2': take_profit2,
                'take_profit3': take_profit3,
                'current_price': current_price,
                'rsi': current_rsi,
                'stochastic': {'k': current_stoch_k, 'd': current_stoch_d},
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                'momentum_score': momentum_score,
                'entry_conditions': entry_conditions,
                'entry_adjustments': entry_adjustments,
                'pivot_points': pivot_points,
                'atr': current_atr,
                'signal_explanation': signal_explanation,
                'timeframe': timeframe_info,
                'indicators': {
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'ema_9': ema_9,
                    'ema_21': ema_21,
                    'rsi': rsi,
                    'stoch_k': stoch_k,
                    'stoch_d': stoch_d,
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'macd': macd_line,
                    'signal': signal_line,
                    'histogram': histogram,
                    'closes': closes,
                    'volumes': volumes,
                    'timestamps': [k['timestamp'] for k in klines],
                    'support_resistance': support_resistance
                }
            }
        
        return None

class ChartGenerator:
    """Generate charts for signals"""
    
    def __init__(self):
        # Thread lock for matplotlib operations
        self.lock = threading.Lock()

    def create_signal_chart(self, symbol, signal_data):
        """Create a chart with indicators and signal - thread safe"""
        with self.lock:
            logger.info(f"Creating chart for {symbol} {signal_data['signal']} signal")
            
            try:
                indicators = signal_data['indicators']
                
                # Create figure with subplots
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12),
                                                        gridspec_kw={'height_ratios': [3, 1, 1, 1]})
                
                # Convert timestamps to dates
                dates = [datetime.fromtimestamp(ts/1000) for ts in indicators['timestamps']]
                
                # Plot 1: Price and moving averages
                ax1.plot(dates, indicators['closes'], 'b-', label='Price', linewidth=2)
                
                # Plot moving averages (filter None values)
                if any(indicators['ema_9']):
                    ema9_filtered = [(date, val) for date, val in zip(dates, indicators['ema_9']) if val is not None]
                    if ema9_filtered:
                        dates_ema9, vals_ema9 = zip(*ema9_filtered)
                        ax1.plot(dates_ema9, vals_ema9, 'orange', label='EMA 9', alpha=0.7)
                
                if any(indicators['ema_21']):
                    ema21_filtered = [(date, val) for date, val in zip(dates, indicators['ema_21']) if val is not None]
                    if ema21_filtered:
                        dates_ema21, vals_ema21 = zip(*ema21_filtered)
                        ax1.plot(dates_ema21, vals_ema21, 'purple', label='EMA 21', alpha=0.7)
                
                if any(indicators['sma_20']):
                    sma20_filtered = [(date, val) for date, val in zip(dates, indicators['sma_20']) if val is not None]
                    if sma20_filtered:
                        dates_sma20, vals_sma20 = zip(*sma20_filtered)
                        ax1.plot(dates_sma20, vals_sma20, 'g-', label='SMA 20', alpha=0.7)
                
                if any(indicators['sma_50']):
                    sma50_filtered = [(date, val) for date, val in zip(dates, indicators['sma_50']) if val is not None]
                    if sma50_filtered:
                        dates_sma50, vals_sma50 = zip(*sma50_filtered)
                        ax1.plot(dates_sma50, vals_sma50, 'r-', label='SMA 50', alpha=0.7)
                
                # Bollinger Bands
                if any(indicators['bb_upper']) and any(indicators['bb_lower']):
                    bb_filtered = [(date, upper, lower) for date, upper, lower in 
                                 zip(dates, indicators['bb_upper'], indicators['bb_lower']) 
                                 if upper is not None and lower is not None]
                    if bb_filtered:
                        dates_bb, upper_bb, lower_bb = zip(*bb_filtered)
                        ax1.fill_between(dates_bb, upper_bb, lower_bb,
                                        alpha=0.2, color='gray', label='Bollinger Bands')
                        
                        if any(indicators['bb_middle']):
                            middle_filtered = [(date, val) for date, val in zip(dates, indicators['bb_middle']) if val is not None]
                            if middle_filtered:
                                dates_middle, vals_middle = zip(*middle_filtered)
                                ax1.plot(dates_middle, vals_middle, 'gray', linestyle='--', alpha=0.5)
                
                # Mark support and resistance levels
                for sr in indicators['support_resistance']:
                    level_type, level, touches = sr
                    color = 'green' if level_type == 'support' else 'red'
                    ax1.axhline(y=level, color=color, linestyle=':', alpha=0.5)
                    ax1.text(dates[-1], level, f'{level_type[0].upper()}: {level:.4f} ({touches})',
                            fontsize=8, color=color)
                
                # Mark signal
                signal_color = 'green' if signal_data['signal'] == 'BUY' else 'red'
                ax1.axhline(y=signal_data['current_price'], color=signal_color,
                           linestyle='--', linewidth=2, label=f"{signal_data['signal']} Signal")
                
                # Entry zones
                if signal_data.get('entry_adjustments'):
                    for adj in signal_data['entry_adjustments']:
                        if 'price' in adj:
                            ax1.axhline(y=adj['price'], color='blue', linestyle=':',
                                       alpha=0.7, label=f"{adj['type']}: {adj['price']:.4f}")
                
                # Stop loss and take profit levels
                ax1.axhline(y=signal_data['stop_loss'], color='red', linestyle='--',
                           label=f"SL: {signal_data['stop_loss']:.4f}")
                ax1.axhline(y=signal_data['take_profit1'], color='green', linestyle=':',
                           label=f"TP1: {signal_data['take_profit1']:.4f}")
                ax1.axhline(y=signal_data['take_profit2'], color='green', linestyle=':',
                           alpha=0.7, label=f"TP2: {signal_data['take_profit2']:.4f}")
                ax1.axhline(y=signal_data['take_profit3'], color='green', linestyle=':',
                           alpha=0.5, label=f"TP3: {signal_data['take_profit3']:.4f}")
                
                ax1.set_title(f"{symbol} - {signal_data['signal']} Signal (Strength: {signal_data['strength']}) | Momentum: {signal_data['momentum_score']} | TF: {signal_data['timeframe']}")
                ax1.set_ylabel('Price')
                ax1.legend(loc='upper left', fontsize=8)
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: RSI & Stochastic
                if any(indicators['rsi']):
                    rsi_filtered = [(date, val) for date, val in zip(dates, indicators['rsi']) if val is not None]
                    if rsi_filtered:
                        dates_rsi, vals_rsi = zip(*rsi_filtered)
                        ax2.plot(dates_rsi, vals_rsi, 'purple', linewidth=2, label='RSI')
                
                if any(indicators['stoch_k']):
                    stochk_filtered = [(date, val) for date, val in zip(dates, indicators['stoch_k']) if val is not None]
                    if stochk_filtered:
                        dates_stochk, vals_stochk = zip(*stochk_filtered)
                        ax2.plot(dates_stochk, vals_stochk, 'blue', linewidth=1, label='Stoch %K')
                
                if any(indicators['stoch_d']):
                    stochd_filtered = [(date, val) for date, val in zip(dates, indicators['stoch_d']) if val is not None]
                    if stochd_filtered:
                        dates_stochd, vals_stochd = zip(*stochd_filtered)
                        ax2.plot(dates_stochd, vals_stochd, 'red', linewidth=1, label='Stoch %D')
                
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax2.axhline(y=80, color='r', linestyle=':', alpha=0.3)
                ax2.axhline(y=20, color='g', linestyle=':', alpha=0.3)
                ax2.fill_between(dates, 30, 70, alpha=0.1, color='gray')
                ax2.set_ylabel('RSI/Stoch')
                ax2.set_ylim(0, 100)
                ax2.legend(loc='upper left', fontsize=8)
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: MACD - FIXED
                if any(indicators['macd']):
                    macd_filtered = [(date, val) for date, val in zip(dates, indicators['macd']) if val is not None]
                    if macd_filtered:
                        dates_macd, vals_macd = zip(*macd_filtered)
                        ax3.plot(dates_macd, vals_macd, 'blue', linewidth=1.5, label='MACD')
                
                if any(indicators['signal']):
                    signal_filtered = [(date, val) for date, val in zip(dates, indicators['signal']) if val is not None]
                    if signal_filtered:
                        dates_signal, vals_signal = zip(*signal_filtered)
                        ax3.plot(dates_signal, vals_signal, 'red', linewidth=1.5, label='Signal')
                
                # Histogram - FIXED
                if any(indicators['histogram']):
                    hist_filtered = [(date, val) for date, val in zip(dates, indicators['histogram']) if val is not None]
                    if hist_filtered:
                        dates_hist, vals_hist = zip(*hist_filtered)
                        colors_hist = ['g' if h > 0 else 'r' for h in vals_hist]
                        ax3.bar(dates_hist, vals_hist, color=colors_hist, alpha=0.3, label='Histogram', width=0.8)
                
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.set_ylabel('MACD')
                ax3.legend(loc='upper left', fontsize=8)
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Volume - FIXED
                colors = ['g' if indicators['closes'][i] > indicators['closes'][i-1] else 'r'
                         for i in range(1, len(indicators['closes']))]
                colors.insert(0, 'g')
                ax4.bar(dates, indicators['volumes'], color=colors, alpha=0.7, width=0.8)
                
                # Volume SMA - FIXED
                vol_sma = TechnicalIndicators.sma(indicators['volumes'], 20)
                if any(vol_sma):
                    vol_sma_filtered = [(date, val) for date, val in zip(dates, vol_sma) if val is not None]
                    if vol_sma_filtered:
                        dates_vol, vals_vol = zip(*vol_sma_filtered)
                        ax4.plot(dates_vol, vals_vol, 'blue', linewidth=2, label='Volume SMA(20)')
                
                ax4.set_ylabel('Volume')
                ax4.set_xlabel('Time')
                ax4.legend(loc='upper left', fontsize=8)
                ax4.grid(True, alpha=0.3)
                
                # Format x-axis
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                # Add text boxes with analysis
                analysis_text = "Entry Conditions:\n" + "\n".join(f"• {c}" for c in signal_data['entry_conditions'][:4])
                reasons_text = "Signal Reasons:\n" + "\n".join(f"• {r}" for r in signal_data['signal_explanation'][:4])
                
                fig.text(0.02, 0.02, analysis_text, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
                fig.text(0.52, 0.02, reasons_text, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
                
                plt.tight_layout()
                
                # Save to BytesIO
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)  # Important: close the figure to free memory
                
                img_buffer.seek(0)
                return img_buffer
                
            except Exception as e:
                logger.error(f"Error creating chart: {e}")
                logger.error(traceback.format_exc())
                plt.close('all')  # Clean up any open figures
                return None

class TelegramNotifier:
    """Send notifications to Telegram"""
    
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.session = requests.Session()

    def send_message(self, text, parse_mode='HTML'):
        """Send text message to Telegram"""
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode
        }
        
        try:
            response = self.session.post(url, json=data, timeout=10)
            if response.status_code == 200:
                logger.info("Message sent to Telegram successfully")
                return True
            else:
                logger.error(f"Failed to send message: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending message to Telegram: {e}")
            return False

    def send_photo(self, photo_buffer, caption=""):
        """Send photo to Telegram"""
        url = f"{self.base_url}/sendPhoto"
        files = {'photo': ('chart.png', photo_buffer, 'image/png')}
        data = {
            'chat_id': self.chat_id,
            'caption': caption[:1024],  # Telegram caption limit
            'parse_mode': 'HTML'
        }
        
        try:
            response = self.session.post(url, data=data, files=files, timeout=30)
            if response.status_code == 200:
                logger.info("Photo sent to Telegram successfully")
                return True
            else:
                logger.error(f"Failed to send photo: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending photo to Telegram: {e}")
            return False

class TradeManager:
    """Manages trading execution and position checking."""

    def __init__(self, exchange_client, api_key=None, secret_key=None):
        self.exchange_client = exchange_client
        self.api_key = api_key
        self.secret_key = secret_key
        logger.info(f"TradeManager initialized with {self.exchange_client.__class__.__name__}.")

    def check_open_positions(self, symbol):
        """Placeholder for checking open positions for a given symbol."""
        # In a real implementation, this would query the exchange API
        # using self.exchange_client and potentially self.api_key/self.secret_key
        logger.info(f"TradeManager: Checking open positions for {symbol} on {self.exchange_client.__class__.__name__}.")
        # Example:
        # positions = self.exchange_client.get_positions(symbol=symbol, api_key=self.api_key, secret_key=self.secret_key)
        # return positions
        return [] # Placeholder: No open positions

    def execute_trade(self, symbol, signal_data, current_klines):
        """Placeholder for executing a trade based on signal_data."""
        # current_klines is passed for potential future use in more advanced execution logic
        # (e.g., checking for immediate price changes, slippage, etc.)
        logger.info(f"TradeManager: Received signal to {signal_data['signal']} {symbol} at {signal_data['current_price']:.4f} on {self.exchange_client.__class__.__name__}.")

        # Example logic (all commented out as it's placeholder):
        # if self.check_open_positions(symbol):
        #     logger.info(f"TradeManager: Already have an open position for {symbol}. Skipping new trade.")
        #     return False

        # order_type = "LIMIT" # or "MARKET"
        # quantity = self.calculate_position_size(signal_data['current_price'], signal_data['stop_loss'])

        # if quantity <= 0:
        #     logger.warning(f"TradeManager: Calculated quantity for {symbol} is zero or negative. Skipping trade.")
        #     return False

        # try:
        #     trade_result = self.exchange_client.place_order(
        #         symbol=symbol,
        #         side=signal_data['signal'], # "BUY" or "SELL"
        #         order_type=order_type,
        #         qty=quantity,
        #         price=signal_data['entry'], # For limit orders
        #         stop_loss=signal_data['stop_loss'],
        #         take_profit=signal_data['take_profit1'],
        #         api_key=self.api_key,
        #         secret_key=self.secret_key
        #     )
        #     if trade_result and trade_result.get('success'): # Structure depends on exchange client's response
        #         logger.info(f"TradeManager: Successfully executed {signal_data['signal']} for {symbol} of {quantity} at {signal_data['entry']}.")
        #         # Store trade details, send notification, etc.
        #         return True
        #     else:
        #         logger.error(f"TradeManager: Failed to execute trade for {symbol}. Response: {trade_result}")
        #         return False
        # except Exception as e:
        #     logger.error(f"TradeManager: Exception during trade execution for {symbol}: {e}")
        #     return False

        logger.info(f"TradeManager: Placeholder for {signal_data['signal']} {symbol}. No actual trade executed.")
        return True # Placeholder: assume success for now

    # def calculate_position_size(self, entry_price, stop_loss_price, risk_per_trade=0.01, account_balance=1000):
    #     """Placeholder for calculating position size."""
    #     # This is a very simplified example. Real position sizing is complex.
    #     # risk_per_trade = 1% of account_balance
    #     # account_balance would need to be fetched or configured
    #     risk_amount = account_balance * risk_per_trade
    #     price_difference = abs(entry_price - stop_loss_price)
    #     if price_difference == 0:
    #         return 0
    #     position_size_in_asset = risk_amount / price_difference
    #     # Convert to contract quantity if needed (e.g. for BTC, 1 contract might be 0.001 BTC)
    #     # This depends on the exchange's contract specifications.
    #     return position_size_in_asset


class CryptoTradingBot:
    """Main trading bot class"""
    
    def __init__(self):
        self.telegram = TelegramNotifier(TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID)
        self.chart_generator = ChartGenerator()
        self.strategy = TradingStrategy()
        self.symbols_queue = queue.Queue()

        self.exchange_client = None
        self.bingx_client = None
        self.bybit_client = None
        self.trade_manager = None # Initialize trade_manager attribute

        logger.info("Initializing exchange clients...")

        # Try BingX first
        if BINGX_API_KEY:
            try:
                logger.info("Attempting to initialize BingX client...")
                self.bingx_client = BingXClient(api_key=BINGX_API_KEY, secret_key=BINGX_SECRET_KEY)
                # Perform a simple test call, e.g., get_symbols, to ensure it's working
                # For now, we'll assume it initializes if keys are present.
                # A proper test would involve making a lightweight API call here.
                self.exchange_client = self.bingx_client
                logger.info("Successfully initialized BingX client.")
            except Exception as e:
                logger.error(f"Failed to initialize BingX client: {e}")
                self.bingx_client = None # Ensure it's None if failed

        # Fallback to Bybit if BingX is not initialized
        if not self.exchange_client:
            try:
                logger.info("Attempting to initialize Bybit client as fallback...")
                self.bybit_client = BybitClient()
                # BybitClient current public methods don't require keys, so it should generally init
                self.exchange_client = self.bybit_client
                logger.info("Successfully initialized Bybit client.")
            except Exception as e:
                logger.error(f"Failed to initialize Bybit client: {e}")
                self.bybit_client = None # Ensure it's None if failed

        if self.exchange_client:
            if self.exchange_client == self.bingx_client:
                logger.info("Active exchange client: BingX")
                try:
                    self.trade_manager = TradeManager(exchange_client=self.exchange_client,
                                                      api_key=BINGX_API_KEY, secret_key=BINGX_SECRET_KEY)
                    logger.info("TradeManager initialized for BingX client.")
                except Exception as e:
                    logger.error(f"Failed to initialize TradeManager for BingX: {e}")
                    self.trade_manager = None
            elif self.exchange_client == self.bybit_client:
                logger.info("Active exchange client: Bybit")
                try:
                    self.trade_manager = TradeManager(exchange_client=self.exchange_client,
                                                      api_key=BYBIT_API_KEY, secret_key=BYBIT_SECRET_KEY)
                    logger.info("TradeManager initialized for Bybit client.")
                except Exception as e:
                    logger.error(f"Failed to initialize TradeManager for Bybit: {e}")
                    self.trade_manager = None

            if not self.trade_manager:
                 logger.warning("TradeManager could not be initialized. Auto-trading features will be disabled.")

        else:
            logger.critical("CRITICAL: No exchange client could be initialized. Bot cannot fetch market data. TradeManager not initialized.")
            # Depending on desired behavior, could raise an error or sys.exit here

        self.signals_sent = {}  # Track sent signals to avoid duplicates
        self.running = True
        self.threads = []
        self.num_workers = 5  # Number of worker threads
        self.signal_lock = threading.Lock()  # Lock for signal sending

    def worker(self):
        """Worker thread to process symbols"""
        if not self.exchange_client:
            logger.error(f"[{threading.current_thread().name}] No exchange client available. Worker thread stopping.")
            return

        while self.running:
            try:
                # Get symbol from queue with timeout
                try:
                    symbol = self.symbols_queue.get(timeout=1)
                except queue.Empty:
                    continue

                logger.info(f"[{threading.current_thread().name}] Scanning {symbol} using {type(self.exchange_client).__name__}...")

                # Get kline data
                klines = self.exchange_client.get_klines(symbol)
                if not klines:
                    logger.warning(f"No kline data for {symbol}")
                    continue

                # Analyze with strategy
                signal_data = self.strategy.analyze(klines)
                
                if signal_data:
                    if self.trade_manager:
                        logger.info(f"Handing off signal for {symbol} to TradeManager.")
                        self.trade_manager.check_open_positions(symbol) # Placeholder call
                        self.trade_manager.execute_trade(symbol, signal_data, current_klines=klines) # Placeholder call
                    else:
                        logger.warning("TradeManager not available, skipping trade execution logic.")

                    # Notification logic (moved slightly to be after trade manager calls)
                    with self.signal_lock:
                        signal_key = f"{symbol}_{signal_data['signal']}"
                        last_sent = self.signals_sent.get(signal_key, 0)
                        current_time = time.time()
                        
                        if current_time - last_sent > 3600:  # 1 hour cooldown
                            logger.info(f"🚨 NOTIFICATION for {symbol}: {signal_data['signal']}")
                            self.send_signal(symbol, signal_data) # This method sends Telegram notification
                            self.signals_sent[signal_key] = current_time
                        else:
                            logger.info(f"Notification for {symbol} already sent recently, skipping...")
                else:
                    logger.debug(f"No signal for {symbol}")

            except Exception as e:
                logger.error(f"Error in worker thread: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1)

    def send_signal(self, symbol, signal_data):
        """Send signal to Telegram with chart"""
        try:
            # Create message
            emoji = "🟢" if signal_data['signal'] == "BUY" else "🔴"
            
            # Format entry adjustments
            entry_text = ""
            if signal_data.get('entry_adjustments'):
                entry_text = "\nFine-tuned Entries:\n"
                for adj in signal_data['entry_adjustments'][:3]:  # Limit to 3
                    entry_text += f"• {adj['reason']}: ${adj.get('price', signal_data['entry']):.4f}\n"

            # Format signal explanation
            explanation_text = "\n📊 Signal Analysis:\n"
            for explanation in signal_data.get('signal_explanation', []):
                explanation_text += f"{explanation}\n"

            message = f"""
{emoji} {signal_data['signal']} Signal - {symbol} {emoji}

🎯 Timeframe: {signal_data['timeframe']} | Strength: {signal_data['strength']}/5 | Momentum: {signal_data['momentum_score']}/4

💰 Current Price: ${signal_data['current_price']:.4f}

📈 Trade Setup:
• Entry Zone: ${signal_data['entry'] * 0.999:.4f} - ${signal_data['entry'] * 1.001:.4f}
• Stop Loss: ${signal_data['stop_loss']:.4f} ({abs((signal_data['stop_loss'] - signal_data['entry']) / signal_data['entry'] * 100):.1f}%)
• Target 1: ${signal_data['take_profit1']:.4f} ({abs((signal_data['take_profit1'] - signal_data['entry']) / signal_data['entry'] * 100):.1f}%)
• Target 2: ${signal_data['take_profit2']:.4f} ({abs((signal_data['take_profit2'] - signal_data['entry']) / signal_data['entry'] * 100):.1f}%)
• Target 3: ${signal_data['take_profit3']:.4f} ({abs((signal_data['take_profit3'] - signal_data['entry']) / signal_data['entry'] * 100):.1f}%)

⚖️ Risk Management:
• Risk/Reward: 1:{abs((signal_data['take_profit1'] - signal_data['entry']) / (signal_data['entry'] - signal_data['stop_loss'])):.1f}
• Position Size: Use 1-2% of capital
• ATR: ${signal_data.get('atr', 0):.4f}

📊 Indicators:
• RSI: {signal_data['rsi']:.2f}
• Stochastic: K={signal_data['stochastic']['k']:.2f}, D={signal_data['stochastic']['d']:.2f}
• Volume: {signal_data['volume_ratio']:.2f}x average

{explanation_text}

{entry_text}

🔧 Pivot Points:
• R1: ${signal_data['pivot_points']['r1']:.4f} | S1: ${signal_data['pivot_points']['s1']:.4f}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""

            # Generate chart
            chart_buffer = self.chart_generator.create_signal_chart(symbol, signal_data)
            
            if chart_buffer:
                # Send to Telegram
                success = self.telegram.send_photo(chart_buffer, caption=message)
                
                if success:
                    logger.info(f"Signal sent successfully for {symbol}")
                    
                    # Send follow-up message with entry conditions
                    conditions_msg = f"⚡ Entry Conditions for {symbol}:\n\n"
                    for i, condition in enumerate(signal_data['entry_conditions'], 1):
                        conditions_msg += f"{i}. {condition}\n"
                    conditions_msg += "\n⚠️ Wait for all conditions to be met before entering the trade!"
                    
                    self.telegram.send_message(conditions_msg)
                else:
                    # Fallback to text-only message
                    self.telegram.send_message(message)
            else:
                # Send text-only if chart generation failed
                self.telegram.send_message(message)
                
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
            logger.error(traceback.format_exc())

    def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Crypto Trading Bot...")

        if not self.exchange_client:
            logger.critical("No exchange client initialized at startup. Bot cannot run scan cycles.")
            self.telegram.send_message("⚠️ Bot Error: No exchange client initialized. Cannot fetch data. Please check configuration and API keys.")
            return # Exit run method if no client

        # Send startup message
        startup_message = (
            "🤖 Crypto Trading Bot Started!\n\n"
            f"✅ Active Exchange: {type(self.exchange_client).__name__}\n"
            "✅ Advanced entry fine-tuning enabled\n"
            "✅ Multi-indicator confirmation\n"
            "✅ Dynamic risk management\n"
            "✅ Enhanced signal explanations\n\n"
            f"Scanning {type(self.exchange_client).__name__} USDT futures for high-probability setups..."
        )
        self.telegram.send_message(startup_message)

        # Start worker threads
        logger.info(f"Starting {self.num_workers} worker threads for {type(self.exchange_client).__name__}...")
        for i in range(self.num_workers):
            thread = threading.Thread(target=self.worker, name=f"Worker-{i+1}")
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

        # Main loop
        scan_interval = 300  # 5 minutes between full scans
        while self.running:
            try:
                logger.info("=" * 50)
                logger.info("Starting new scan cycle...")

                if not self.exchange_client:
                    logger.critical("No exchange client available. Cannot proceed with scan cycle.")
                    time.sleep(60) # Wait before retrying or exiting
                    continue

                # Get all symbols
                symbols = self.exchange_client.get_symbols()
                if not symbols: # Handles None or empty list
                    logger.error(f"Failed to get symbols from {type(self.exchange_client).__name__}, waiting before retry...")
                    time.sleep(60)
                    continue

                # Clear the queue
                while not self.symbols_queue.empty():
                    try:
                        self.symbols_queue.get_nowait()
                    except queue.Empty:
                        break

                # Add symbols to queue
                logger.info(f"Adding {len(symbols)} symbols to queue...")
                for symbol in symbols:
                    self.symbols_queue.put(symbol)

                # Wait for queue to be processed
                logger.info("Processing symbols...")
                start_time = time.time()
                while not self.symbols_queue.empty() and time.time() - start_time < scan_interval:
                    queue_size = self.symbols_queue.qsize()
                    if queue_size > 0:
                        logger.info(f"Queue size: {queue_size} symbols remaining...")
                    time.sleep(10)

                logger.info("Scan cycle completed, waiting for next cycle...")

                # Clean up old signals
                current_time = time.time()
                self.signals_sent = {k: v for k, v in self.signals_sent.items()
                                   if current_time - v < 7200}  # Keep 2 hours

                # Wait before next scan
                remaining_time = max(0, scan_interval - (time.time() - start_time))
                if remaining_time > 0:
                    logger.info(f"Waiting {remaining_time:.0f} seconds before next scan...")
                    time.sleep(remaining_time)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(30)

        # Cleanup
        self.running = False
        logger.info("Waiting for worker threads to finish...")
        for thread in self.threads:
            thread.join(timeout=5)
        logger.info("Bot stopped.")

def main():
    """Main entry point"""
    logger.info("Initializing Crypto Trading Bot with Advanced Entry Fine-tuning...")
    
    # Test Telegram connection
    telegram = TelegramNotifier(TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID)
    if not telegram.send_message("🔧 Bot Initialization\n\nTesting Telegram connection..."):
        logger.error("Failed to connect to Telegram. Please check your API credentials.")
        return

    # Create and run bot
    bot = CryptoTradingBot()
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        telegram.send_message(f"❌ Bot Crashed!\n\nError: {str(e)}")

if __name__ == "__main__":
    main()

