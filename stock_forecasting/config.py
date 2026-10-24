"""Load settings from config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass
class Config:
    data_path: Path
    tickers: list[str]
    history_days: int = 730
    initial_train_days: int = 250
    refit_every_days: int = 21
    models: list[str] = field(default_factory=lambda: ["naive", "ridge", "arima", "lstm", "transformer"])
    window: int = 7
    seed: int = 42
    model_params: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = tomllib.loads(path.read_text())
    data, bt, models = raw["data"], raw["backtest"], raw["models"]
    data_path = Path(data["path"])
    if not data_path.is_absolute():
        data_path = path.parent / data_path
    return Config(
        data_path=data_path,
        tickers=[t.upper() for t in data["tickers"]],
        history_days=data.get("history_days", 730),
        initial_train_days=bt["initial_train_days"],
        refit_every_days=bt["refit_every_days"],
        models=bt["models"],
        window=models.get("window", 7),
        seed=models.get("seed", 42),
        model_params={k: v for k, v in models.items() if isinstance(v, dict)},
        training=raw.get("training", {}),
    )
