# Binance Futures Trading Bot

A sophisticated, high win-rate trading bot for Binance USDT-M Futures with advanced algorithmic order execution.

## Features

### AMR-VF Strategy (Adaptive Momentum Reversion with Volatility Filtering)
- **High Win Rate Design**: 60-70% target win rate through multiple confluence factors
- **Low Drawdown**: <15% max drawdown through strict risk management
- **Multi-Timeframe Analysis**: Confirms trends across 15m, 1h, and 4h timeframes
- **Mean Reversion Entries**: Enter on pullbacks within trending markets
- **Volatility Regime Filtering**: Avoids trading in unfavorable conditions

### Binance Algo Order API Integration
- **TWAP Orders**: Time-Weighted Average Price for large orders (minimum $1,000)
- **VP Orders**: Volume Participation for optimal execution
- **Smart Order Routing**: Automatically selects best execution method
- **Conditional Orders**: Uses new `/fapi/v1/algoOrder` endpoint for stops/TPs

### Risk Management
- **Dynamic Position Sizing**: Based on ATR and signal strength
- **Daily Drawdown Protection**: Stops trading after reaching max daily loss
- **Consecutive Loss Handling**: Reduces size or pauses after loss streaks
- **Maximum Exposure Limits**: Caps total portfolio exposure
- **Trailing Stops**: Lock in profits as trades move favorably

### Trade Execution
- **Scaled Exits**: Three take-profit levels (40%/30%/30% distribution)
- **ATR-Based Stops**: Dynamic stop loss based on volatility
- **Support/Resistance Integration**: Adjusts levels to key price zones
- **Paper Trading Mode**: Test strategies without risking capital

## Installation

```bash
# Clone or download the repository
cd binance-trading-bot

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API credentials
nano .env
```

## Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Binance API (required for live trading)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=false

# Trading Mode: paper (default) or live
TRADING_MODE=paper

# Risk Settings
MAX_POSITION_SIZE_PCT=2.0   # Max 2% per position
MAX_DAILY_LOSS_PCT=5.0      # Stop after 5% daily loss
DEFAULT_RISK_PCT=1.0        # Risk 1% per trade
DEFAULT_LEVERAGE=10         # 10x leverage

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### API Key Permissions

Your Binance API key needs:
- ✅ Enable Futures
- ✅ Enable Spot & Margin Trading (for SAPI endpoints)
- ❌ Enable Withdrawals (NOT needed)

## Usage

### Paper Trading (Recommended for Testing)
```bash
python run.py --mode paper
```

### Live Trading
```bash
python run.py --mode live
```

### Using Testnet
```bash
python run.py --testnet
```

### Check Configuration
```bash
python run.py --status
```

## Project Structure

```
├── config.py           # Configuration dataclasses
├── binance_client.py   # Binance API client with algo orders
├── strategy.py         # AMR-VF trading strategy
├── risk_manager.py     # Risk management system
├── order_executor.py   # Order execution (live & paper)
├── trading_bot.py      # Main bot orchestrator
├── run.py             # CLI runner script
├── requirements.txt    # Python dependencies
└── .env.example       # Environment template
```

## Trading Strategy Details

### Entry Conditions (Confluence Factors)

The strategy requires multiple factors to align for a trade:

1. **Trend Alignment** (2 points): Trade with higher timeframe trend
2. **RSI Condition** (1-2 points): Oversold for longs, overbought for shorts
3. **Stochastic Cross** (1 point): Bullish/bearish cross in extreme zones
4. **MACD Momentum** (1 point): Histogram confirming direction
5. **Support/Resistance** (1 point): Price near key levels
6. **Bollinger Band** (1 point): Price at band extremes
7. **Volume Confirmation** (1 point): Above average volume
8. **Candlestick Pattern** (1 point): Hammer, engulfing, etc.

**Minimum Required: 4/10 confluence points**

### Exit Strategy

- **TP1**: 2x ATR (close 40%)
- **TP2**: 3x ATR (close 30%)
- **TP3**: 5x ATR (close 30%)
- **Stop Loss**: 1.5x ATR (or below support/resistance)

### Volatility Regime Filter

Skips trading when:
- **Low Volatility**: ATR < 50% of average (consolidation)
- **Extreme Volatility**: ATR > 200% of average (unpredictable)

Only trades in **Normal** volatility conditions.

## Algo Order API

The bot uses Binance's new algo order endpoints:

### TWAP (Time-Weighted Average Price)
```
POST /sapi/v1/algo/futures/newOrderTwap

Used for orders > $5,000 to minimize market impact
- Duration: 5 minutes to 24 hours
- Splits order into smaller chunks over time
```

### Conditional Orders (Stop/TP)
```
POST /fapi/v1/algoOrder

Required since 2025-12-09 migration for:
- STOP_MARKET
- TAKE_PROFIT_MARKET
- TRAILING_STOP_MARKET
```

## Risk Management Rules

| Rule | Default | Description |
|------|---------|-------------|
| Max Position Size | 2% | Maximum single position size |
| Max Daily Loss | 5% | Stop trading after daily loss |
| Max Total Exposure | 20% | Maximum total portfolio exposure |
| Max Trades/Day | 10 | Daily trade limit |
| Max Concurrent | 5 | Maximum simultaneous positions |
| Min Risk/Reward | 1:2 | Minimum R:R ratio required |
| Consecutive Loss Limit | 5 | Pause after 5 losses |

## Notifications

Telegram notifications include:
- 🟢 Long signals with entry/exit levels
- 🔴 Short signals with entry/exit levels
- ✅ Trade opened confirmations
- 💰 Trade closed with P&L
- 📊 Daily trading summaries

## Disclaimer

⚠️ **TRADING INVOLVES SIGNIFICANT RISK**

- This bot is for educational purposes
- Past performance doesn't guarantee future results
- Always test with paper trading first
- Never risk more than you can afford to lose
- The authors are not responsible for any losses

## License

MIT License - Use at your own risk.
