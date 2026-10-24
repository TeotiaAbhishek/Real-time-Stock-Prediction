"""Command line entry point: ``stock-forecast download | backtest | predict``."""

from __future__ import annotations

import argparse

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stock-forecast")
    parser.add_argument("--config", help="path to config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download fresh prices from Yahoo Finance into the data path")
    bt = sub.add_parser("backtest", help="walk-forward backtest of every model")
    bt.add_argument("--models", nargs="+", help="subset of models, e.g. naive ridge arima")
    bt.add_argument("--tickers", nargs="+")
    bt.add_argument("--out", default="reports")
    pr = sub.add_parser("predict", help="forecast the next trading day")
    pr.add_argument("--models", nargs="+")
    pr.add_argument("--tickers", nargs="+")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "download":
        from .data import download_prices

        prices = download_prices(config.tickers, config.history_days)
        config.data_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(config.data_path)
        print(f"Saved {len(prices)} days to {config.data_path}")
    elif args.command == "backtest":
        from .backtest import run_backtest, score, summarise
        from .report import write_reports

        preds = run_backtest(config, args.models, args.tickers)
        per_stock = score(preds)
        summary = summarise(per_stock)
        write_reports(preds, per_stock, summary, args.out)
        print(summary.round(4).to_string(index=False))
    elif args.command == "predict":
        from .predict import forecast_next_day

        print(forecast_next_day(config, args.models, args.tickers).to_string(index=False))


if __name__ == "__main__":
    main()
