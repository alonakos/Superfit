## Superfit

<img width="1910" height="945" alt="sfgui" src="https://github.com/user-attachments/assets/dd99d63f-fbdd-439f-9937-2f0e5dae05d8" />
<img width="1910" height="943" alt="sggui" src="https://github.com/user-attachments/assets/8e84d2be-f8c6-4ba5-a463-2d9a8c3662fb" />


## Quick Start

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

---

## License


## Contact

