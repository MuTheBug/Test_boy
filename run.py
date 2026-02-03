#!/usr/bin/env python3
"""
Binance Futures Trading Bot - Runner Script

Usage:
    python run.py                     # Run in paper trading mode (default)
    python run.py --mode live         # Run in live trading mode
    python run.py --mode paper        # Run in paper trading mode
    python run.py --testnet           # Use Binance testnet
    python run.py --status            # Show bot status
"""

import os
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars


def setup_environment(args):
    """Setup environment variables from command line arguments"""
    if args.mode:
        os.environ['TRADING_MODE'] = args.mode

    if args.testnet:
        os.environ['BINANCE_TESTNET'] = 'true'

    if args.api_key:
        os.environ['BINANCE_API_KEY'] = args.api_key

    if args.api_secret:
        os.environ['BINANCE_API_SECRET'] = args.api_secret


def validate_config():
    """Validate configuration before starting"""
    mode = os.getenv('TRADING_MODE', 'paper').lower()

    if mode == 'live':
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')

        if not api_key or not api_secret:
            print("ERROR: Live trading requires BINANCE_API_KEY and BINANCE_API_SECRET")
            print("Set these in .env file or as environment variables")
            sys.exit(1)

        print("⚠️  WARNING: Live trading mode enabled!")
        print("Real orders will be placed on Binance Futures")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)


def show_status():
    """Show current bot configuration status"""
    print("\n=== Trading Bot Configuration ===\n")

    mode = os.getenv('TRADING_MODE', 'paper')
    testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
    api_key = os.getenv('BINANCE_API_KEY', '')

    print(f"Trading Mode:  {mode.upper()}")
    print(f"Testnet:       {'Yes' if testnet else 'No'}")
    print(f"API Key:       {'Configured' if api_key else 'Not set'}")

    print(f"\nRisk Settings:")
    print(f"  Max Position Size: {os.getenv('MAX_POSITION_SIZE_PCT', '2.0')}%")
    print(f"  Max Daily Loss:    {os.getenv('MAX_DAILY_LOSS_PCT', '5.0')}%")
    print(f"  Default Risk:      {os.getenv('DEFAULT_RISK_PCT', '1.0')}%")
    print(f"  Default Leverage:  {os.getenv('DEFAULT_LEVERAGE', '10')}x")

    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    print(f"\nTelegram:      {'Configured' if telegram_token else 'Not set'}")

    print("\n================================\n")


def main():
    parser = argparse.ArgumentParser(
        description='Binance Futures Trading Bot with AMR-VF Strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    Start in paper trading mode
  python run.py --mode live        Start in live trading mode
  python run.py --testnet          Use Binance testnet
  python run.py --status           Show configuration status

Environment Variables:
  BINANCE_API_KEY       Your Binance API key
  BINANCE_API_SECRET    Your Binance API secret
  TRADING_MODE          Trading mode (paper/live)
  BINANCE_TESTNET       Use testnet (true/false)
        """
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['paper', 'live'],
        help='Trading mode (default: paper)'
    )

    parser.add_argument(
        '--testnet', '-t',
        action='store_true',
        help='Use Binance testnet'
    )

    parser.add_argument(
        '--api-key',
        help='Binance API key (or set BINANCE_API_KEY env var)'
    )

    parser.add_argument(
        '--api-secret',
        help='Binance API secret (or set BINANCE_API_SECRET env var)'
    )

    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show configuration status and exit'
    )

    parser.add_argument(
        '--backtest', '-b',
        action='store_true',
        help='Run backtest instead of live/paper trading'
    )

    parser.add_argument(
        '--symbol',
        default='BTCUSDT',
        help='Symbol for backtest (default: BTCUSDT)'
    )

    parser.add_argument(
        '--candles',
        type=int,
        default=1000,
        help='Number of candles for backtest (default: 1000)'
    )

    parser.add_argument(
        '--multi-backtest',
        action='store_true',
        help='Run backtest on multiple symbols'
    )

    parser.add_argument(
        '--balance',
        type=float,
        default=5.0,
        help='Initial balance for backtest (default: 5.0 USDT)'
    )

    parser.add_argument(
        '--interval',
        default='15m',
        help='Timeframe for backtest: 5m, 15m, 1h, 4h (default: 15m)'
    )

    args = parser.parse_args()

    # Setup environment
    setup_environment(args)

    if args.status:
        show_status()
        sys.exit(0)

    # Handle backtest mode
    if args.backtest or args.multi_backtest:
        print("\n" + "=" * 50)
        print("  BACKTEST MODE")
        print("  AMR-VF Strategy")
        print("=" * 50)
        print(f"\n  Initial Balance: ${args.balance:.2f}")
        print(f"  Timeframe: {args.interval}")
        print(f"  Candles: {args.candles}\n")

        from backtest import Backtester

        backtester = Backtester(initial_balance=args.balance)

        if args.multi_backtest:
            symbols = ['BTCUSDT', 'ETHUSDT']  # Only profitable pairs
            backtester.run_multi_symbol_backtest(symbols, args.interval, args.candles)
        else:
            result = backtester.run_backtest(args.symbol, args.interval, args.candles)
            backtester.print_results(result)

        sys.exit(0)

    # Validate configuration for live/paper mode
    validate_config()

    # Show startup info
    print("\n" + "=" * 50)
    print("  BINANCE FUTURES TRADING BOT")
    print("  AMR-VF Strategy (High Win Rate, Low Drawdown)")
    print("=" * 50)

    mode = os.getenv('TRADING_MODE', 'paper').upper()
    print(f"\nMode: {mode}")
    print("Starting bot...\n")

    # Import and run bot
    from trading_bot import main as run_bot
    run_bot()


if __name__ == "__main__":
    main()
