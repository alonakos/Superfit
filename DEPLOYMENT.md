# Deployment Guide

## Local Development

### Prerequisites
- Python 3.10
- `uv` package manager (optional but recommended)

### Setup

```bash
git clone git@github.com:alonakos/GUIsuperfit_.git
cd GUISuperfit_

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# OR with uv:
uv pip install .
```

### Run locally
```bash
python GUISuperfit/mergedapp/app.py
```
Open http://127.0.0.1:8050/