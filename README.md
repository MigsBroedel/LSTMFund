# LSTMFund

Fundamental analysis and stock-ranking system: it collects fundamentals and price data, engineers features, detects patterns, and ranks equities — with an LSTM model and a backtester to evaluate the ranking against historical data.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)
![uv](https://img.shields.io/badge/uv-packaging-DE5FE9)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## About

LSTMFund ranks stocks by combining fundamental data with learned patterns rather than a single heuristic. It pulls data via `yfinance`, engineers features, and scores equities into a ranking exported as CSV.

Beyond the ranking entrypoint (`fund.py`), the `patterns/` package is a small quantitative pipeline: a **data collector**, a **feature engineer**, a **pattern detector**, a **hybrid model** (`model_hybrid.py`), a **risk manager**, a **backtester** to validate signals against history, and a **visualizer**. Splitting these responsibilities keeps the modeling honest — signals are backtested and risk-adjusted, not just fitted.

## Stack

- **Python** (managed with **uv**)
- **TensorFlow** — LSTM model
- **scikit-learn**, **NumPy**, **pandas**, **scipy** — features & modeling
- **yfinance** — market/fundamental data
- **matplotlib**, **seaborn** — visualization

## Running locally

```bash
brew install uv          # or: pipx install uv
git clone https://github.com/MigsBroedel/LSTMFund.git
cd LSTMFund
uv sync
uv run python fund.py
```

Generated CSVs are written to `data/processed/`.

## Architecture

```
fund.py                      → ranking entrypoint
patterns/
├── data_collector.py        → market/fundamental data
├── feature_engineer.py      → feature construction
├── pattern_detector.py      → signal detection
├── model_hybrid.py          → LSTM + hybrid scoring
├── risk_manager.py          → position/risk sizing
├── backtester.py            → historical validation
└── visualizer.py            → charts
```

## Known limitations & roadmap

- Educational/research project — not investment advice and not tuned for live trading.
- Roadmap: walk-forward validation, a wider fundamental feature set, and persisted model checkpoints.

## License

MIT
