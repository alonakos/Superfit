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

---

## Deployment on Render

### Prerequisites
- Render.com account
- GitHub repository (push your code)

### Steps

1. **Connect Repository**
   - Go to https://dashboard.render.com/
   - Click "New +" → "Web Service"
   - Connect your GitHub account and select this repository

2. **Configure Service**
   - **Name**: `guisuperfit` (or your preferred name)
   - **Environment**: `Python 3.10`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     gunicorn GUISuperfit.mergedapp.app:server
     ```

3. **Environment Variables** (if needed)
   - Add any environment variables in the dashboard

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy

### Note
The `Procfile` in the root directory defines the startup command. Render will use it automatically if you don't specify a custom start command.

---

## Project Structure

```
GUISuperfit_/
├── GUISuperfit/
│   ├── mergedapp/           # Main Dash application
│   │   ├── app.py           # Entry point (Flask server: app.server)
│   │   ├── pages/           # Dash pages
│   │   ├── assets/          # CSS, images
│   │   └── sne/, gal/, ...  # Data
│   └── multi_object/        # Sample spectrum data
├── NGSF/                    # Superfit backend package
│   ├── NGSF/                # Core package code
│   └── setup.py
├── pyproject.toml           # Project configuration
├── requirements.txt         # Python dependencies
├── Procfile                 # Deployment configuration
└── README.md
```

## Troubleshooting

**Port issues on Render:**
- Render automatically assigns a port via `PORT` environment variable
- The Gunicorn command uses port 8000 by default (which Render manages)

**Dependencies not installing:**
- Ensure `requirements.txt` is in the root directory
- Check for version conflicts in pyproject.toml

**App not starting:**
- Check Render logs in the dashboard
- Verify `GUISuperfit.mergedapp.app:server` path is correct
