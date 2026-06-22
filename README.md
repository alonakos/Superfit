# GUIsuperfit

A modern, responsive Dash interface for running and inspecting Superfit spectrum fitting results with dark mode support.

## Features

✨ **Interactive Interface**
- Intuitive two-page design: Input & Output
- Smooth, responsive data visualizations
- Real-time fitting progress monitoring

🌓 **Dark Mode**
- Light/dark theme toggle
- Optimized for different lighting conditions

📊 **Advanced Spectrum Fitting**
- Upload `.dat` spectrum files
- Configure SN type, redshift, extinction, and other parameters
- Real-time progress tracking with percentage completion
- Visualize fitted results with interactive Plotly charts

---

## Quick Start

### Prerequisites
- Python 3.10+
- Git

### Local Development

```bash
# Clone repository
git clone https://github.com/alonakos/GUIsuperfit_.git
cd GUIsuperfit_

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python GUISuperfit/mergedapp/app.py
```

Open your browser to **http://127.0.0.1:8050/**

### Using UV (Optional - Faster)

```bash
# Install uv
pip install uv
# or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv pip install -r requirements.txt
```

---

## Deployment

### Render (Recommended)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed Render deployment instructions.

Quick summary:
1. Push to GitHub
2. Connect repo to Render
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn GUISuperfit.mergedapp.app:server`

---

## Project Structure

```
GUISuperfit_/
├── GUISuperfit/
│   ├── mergedapp/              # Main Dash application
│   │   ├── app.py              # Entry point
│   │   ├── pages/              # Dash multi-page setup
│   │   │   ├── sfgui.py        # Input page
│   │   │   └── sggui.py        # Output page
│   │   ├── assets/             # CSS, styling
│   │   └── sne/, gal/          # Template data
│   └── multi_object/           # Example spectrum files
├── NGSF/                       # Superfit backend package
│   ├── NGSF/                   # Core fitting modules
│   ├── setup.py
│   └── ...
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project config
├── Procfile                    # Deployment config
├── DEPLOYMENT.md               # Deployment guide
└── README.md
```

## Usage

### Input Page (`/sfgui`)
1. Upload a `.dat` spectrum file
2. Select supernova type (Ia, Ib, Ic, II, SLSN, etc.)
3. Configure parameters:
   - Redshift
   - Reddening (E(B-V))
   - Other fitting parameters
4. Click "Run Fit" to start Superfit
5. Monitor progress with real-time percentage indicator

### Output Page (`/sggui`)
1. View fitted spectrum alongside original data
2. Inspect fit residuals and quality metrics
3. Download results as CSV
4. Adjust parameters and re-run if needed

---

## Technology Stack

- **Backend:** Flask, Superfit
- **Frontend:** Dash, Plotly, Bootstrap
- **Data Processing:** Pandas, NumPy, SciPy, Astropy
- **Deployment:** Gunicorn, Render

## Dependencies

See [requirements.txt](requirements.txt) for full dependency list:
- dash, plotly, dash-bootstrap-components
- astropy, scipy, pandas, numpy
- matplotlib, extinction, pyastronomy
- gunicorn (for deployment)

---

## License

[Add your license here]

## Contact

For issues or questions, please open a GitHub issue.
