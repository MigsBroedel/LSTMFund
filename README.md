# LSTMFund

Fundamental analysis and stock ranking system.

## Setup

Install uv:
```bash
brew install uv
```

Install dependencies and run:
```bash
uv sync
uv run python fund.py
```

## Output

Generated CSV files are saved to `data/processed/`:
- `ranking_fundamentalista.csv` - Complete ranking with scores
- `sinais_ensemble.csv` - Trading signals
- `red_flags_detalhados.csv` - Risk alerts
- `top_picks.csv` - Top recommendations
