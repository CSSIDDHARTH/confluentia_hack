# shelfSense AI

An AI retail compliance agent. It analyses shelf images to find planogram violations, price-label mismatches and stock-outs, and produces a compliance score and corrective actions.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/make_mock_image.py
pytest
streamlit run app/dashboard.py
```

See `CLAUDE.md` for the project layout, data contracts, assumptions and team agreements.
