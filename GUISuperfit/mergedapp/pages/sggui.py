import os
from pathlib import Path

import numpy as np
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, callback
import plotly.graph_objs as go
import extinction
from extinction import apply

dash.register_page(__name__, path="/sggui")

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_NGSF_DIR = PROJECT_ROOT / "NGSF"

NGSF_BASE = Path(os.environ.get("NGSF_DIR", str(DEFAULT_NGSF_DIR))).resolve()
RESULTS_DIR = Path(os.environ.get("NGSF_RESULTS_DIR", str(NGSF_BASE / "results"))).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_COLUMNS = [
    "SPECTRUM", "GALAXY", "SN", "CONST_SN", "CONST_GAL",
    "Z", "A_v", "Phase", "Band",
    "Frac(SN)", "Frac(gal)", "CHI2/dof", "CHI2/dof2", "sn_name",
]
TABLE_COLUMNS = [c for c in RESULT_COLUMNS if c != "sn_name"]

COLUMN_ALIASES = {
    "CONST SN": "CONST_SN",
    "CONST_SN ": "CONST_SN",
    "CONST GAL": "CONST_GAL",
    "CONST_GAL ": "CONST_GAL",
    "Frac SN": "Frac(SN)",
    "Frac_SN": "Frac(SN)",
    "Frac(SN) ": "Frac(SN)",
    "Frac gal": "Frac(gal)",
    "Frac_gal": "Frac(gal)",
    "Frac(gal) ": "Frac(gal)",
    "CHI2 dof": "CHI2/dof",
    "CHI2_dof": "CHI2/dof",
    "CHI2 dof2": "CHI2/dof2",
    "CHI2_dof2": "CHI2/dof2",
}

_PLOT_TOGGLE_OPTIONS = [
    {"label": "Observation − Galaxy", "value": "omg"},
    {"label": "Template", "value": "tem"},
    {"label": "Galaxy", "value": "gal"},
    {"label": "Observation", "value": "obs"},
    {"label": "Normalized Template", "value": "ute"},
]
_ALL_PLOT_VALUES = [opt["value"] for opt in _PLOT_TOGGLE_OPTIONS]

_TRACE_STYLES = {
    "obs": {"color": "#111111"},
    "gal": {"color": "#d62728"},
    "tem": {"color": "#9467bd"},
    "ute": {"color": "#ff7f0e", "dash": "dash"},
    "omg": {"color": "#2ca02c"},
}

_TRACE_STYLES_DARK = {
    "obs": {"color": "#e2e8f0"},
    "gal": {"color": "#ff6b6b"},
    "tem": {"color": "#c77dff"},
    "ute": {"color": "#ffd166", "dash": "dash"},
    "omg": {"color": "#2ec4b6"},
}


# Helpers 
def _under_root(root: Path, p, *extra_roots: Path):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return None
    target = Path(str(p))
    if target.is_absolute():
        return target
    for base in (root, *extra_roots):
        candidate = base / target
        if candidate.exists():
            return candidate
    return root / target


def find_latest_csv_in(folder: Path):
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.csv"), key=lambda q: q.stat().st_mtime, reverse=True)
    return files[0] if files else None


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    for c in RESULT_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    ordered = [c for c in RESULT_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered and c != "sn_name"]
    return df[ordered + extras]


def read_two_col_txt(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, delim_whitespace=True, header=None)
    except Exception:
        df = pd.read_csv(path, header=None)
    df = df.select_dtypes(include="number").iloc[:, :2]
    df.columns = ["wav", "flux"]
    return df.dropna().sort_values("wav").reset_index(drop=True)


def normalize_flux(df: pd.DataFrame) -> pd.DataFrame:
    med = np.nanmedian(df["flux"].to_numpy(dtype=float))
    if not np.isfinite(med) or med == 0:
        med = 1.0
    out = df.copy()
    out["flux"] = out["flux"] / med
    return out


def alam(wavelength, A_v=1.0, R_v=3.1):
    wavelength = np.asarray(wavelength, dtype=float)
    unit_flux = np.ones(wavelength.size, dtype=float)
    return np.asarray(
        apply(extinction.ccm89(wavelength, A_v, R_v), unit_flux),
        dtype=float,
    )


def _fit_grid_component(
    df: pd.DataFrame,
    target_wavelength,
    redshift: float,
    scale: float,
    extinction_magnitude=None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["wav", "flux"])

    z_factor = 1.0 + float(redshift)
    if not np.isfinite(z_factor) or z_factor <= 0:
        z_factor = 1.0

    normalized = normalize_flux(df)
    rest_wavelength = normalized["wav"].to_numpy(dtype=float)
    transformed_flux = normalized["flux"].to_numpy(dtype=float)

    if extinction_magnitude is not None:
        transformed_flux = transformed_flux * 10.0 ** (
            -0.4 * float(extinction_magnitude) * alam(rest_wavelength)
        )

    observed_wavelength = rest_wavelength * z_factor
    observed_flux = transformed_flux / z_factor
    target_wavelength = np.asarray(target_wavelength, dtype=float)

    interpolated_flux = np.interp(
        target_wavelength,
        observed_wavelength,
        observed_flux,
        left=np.nan,
        right=np.nan,
    )

    return pd.DataFrame(
        {
            "wav": target_wavelength,
            "flux": float(scale) * interpolated_flux,
        }
    )


def binspec(df: pd.DataFrame, start_w: float, end_w: float, step: float) -> pd.DataFrame:
    xs = np.arange(start_w, end_w, step, dtype=float)
    ys = np.interp(xs, df["wav"].to_numpy(), df["flux"].to_numpy(), left=np.nan, right=np.nan)
    return pd.DataFrame({"wav": xs, "flux": ys})


def _fmt_float(x, nd=3):
    try:
        v = float(x)
    except Exception:
        return "unknown"
    if np.isfinite(v):
        return f"{v:.{nd}f}"
    return "unknown"


def _fmt_pct(x):
    try:
        v = float(x) * 100.0
    except Exception:
        return "unknown"
    if np.isfinite(v):
        return f"{v:.1f}%"
    return "unknown"


def _base_figure(dark: bool = False) -> go.Figure:
    if dark:
        tmpl   = "plotly_dark"
        kwargs = dict(
            paper_bgcolor="#1a1d27",
            plot_bgcolor="#1a1d27",
            xaxis=dict(title="Wavelength", tickformat=".0f",
                       gridcolor="#2e3347", color="#e2e8f0"),
            yaxis=dict(title="Flux arbitrary",
                       gridcolor="#2e3347", color="#e2e8f0"),
            legend=dict(x=0.88, y=0.98, bgcolor="rgba(26,29,39,0.85)",
                        font=dict(color="#e2e8f0")),
        )
    else:
        tmpl   = "plotly_white"
        kwargs = dict(
            xaxis=dict(title="Wavelength", tickformat=".0f"),
            yaxis=dict(title="Flux arbitrary"),
            legend=dict(x=0.88, y=0.98, bgcolor="rgba(255,255,255,0.6)"),
        )

    fig = go.Figure()
    fig.update_layout(
        template=tmpl,
        margin=dict(l=30, r=20, t=20, b=40),
        showlegend=True,
        uirevision="sggui",
        **kwargs,
    )
    return fig


def _line_style(key: str, dark: bool = False, width: int = 2) -> dict:
    base  = {"width": width}
    extra = (_TRACE_STYLES_DARK if dark else _TRACE_STYLES).get(key, {})
    base.update({k: v for k, v in extra.items() if v is not None})
    return base


def _safe_float(val):
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _empty_results_response():
    return None, None, [], [], []


def _fallback_figure(stored_fig, dark: bool = False):
    if stored_fig:
        fig = go.Figure(stored_fig)
        return fig, fig.to_dict()
    fig = _base_figure(dark)
    return fig, fig.to_dict()

bestfit_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Best Fit", className="mb-0")),
        dbc.CardBody(id="sggui-bestfit"),
    ],
    className="border-success",
    style={"borderWidth": "2px"},
)

bestfit_wrapper = html.Div(
    bestfit_card,
    id="sggui-bestfit-wrapper",
    className="mb-3",
    style={"display": "none"},
)

graph = dcc.Graph(
    id="sggui-graph",
    config={"displayModeBar": False, "responsive": True},
    style={"height": "65vh"},
)

graph_card = dbc.Card(
    dbc.CardBody(graph),
    className="shadow-sm mb-3",
)

options_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Display Options", className="mb-0")),
        dbc.CardBody(
            [
                bestfit_wrapper,
                html.Div(
                    [
                        html.Span("Show graphs:", className="fw-semibold me-2"),
                        dcc.Checklist(
                            id="sggui-plots",
                            options=_PLOT_TOGGLE_OPTIONS,
                            value=_ALL_PLOT_VALUES,
                            inputStyle={"marginRight": "6px"},
                            labelStyle={
                                "display": "inline-block",
                                "marginRight": "18px",
                                "marginBottom": "6px",
                            },
                            persistence=True,
                            persistence_type="session",
                        ),
                    ],
                    className="mb-3",
                ),
                html.Div(
                    [
                        html.Span("Binning (Å):", className="fw-semibold me-2"),
                        dcc.Input(id="sggui-bin", type="number", value=10,
                                  style={"width": "120px"}),
                    ]
                ),
            ]
        ),
    ],
    className="shadow-sm mb-3",
)

results_table = dash_table.DataTable(
    id="sggui-table",
    columns=[{"name": c, "id": c} for c in TABLE_COLUMNS],
    data=[],
    editable=False,
    row_selectable="single",
    selected_rows=[],
    style_header={"fontWeight": "bold", "textAlign": "center"},
    style_cell={"padding": "4px", "fontSize": "12px"},
    style_table={"width": "100%", "overflowX": "auto", "border": "thin lightgrey solid"},
)

table_card = dbc.Card(
    [
        dbc.CardHeader(html.H4("Galaxy Spectrum Match", className="mb-0")),
        dbc.CardBody(results_table),
    ],
    className="shadow-sm mb-4",
)

layout = html.Div(
    [
        dbc.Container(
            [
                options_card,
                graph_card,
                table_card,
                dcc.Store(id="sggui-results-path", storage_type="memory"),
                dcc.Store(id="sggui-results-data", storage_type="memory"),
                dcc.Store(id="sggui-last-figure", storage_type="memory"),
            ],
            fluid=True,
            className="py-3",
        ),
    ]
)


@callback(
    Output("sggui-results-data", "data"),
    Output("sggui-results-path", "data"),
    Output("sggui-table", "data"),
    Output("sggui-table", "columns"),
    Output("sggui-table", "selected_rows"),
    Input("run-flag", "data"),
    prevent_initial_call=False,
)
def load_results(run_flag):
    if isinstance(run_flag, dict) and run_flag.get("action") == "clear":
        return _empty_results_response()

    latest = find_latest_csv_in(RESULTS_DIR)
    if latest is None:
        return _empty_results_response()

    try:
        df = pd.read_csv(latest)
    except Exception:
        return _empty_results_response()

    df = ensure_columns(df)
    sort_cols = [c for c in ["CHI2/dof", "CHI2/dof2"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=True, kind="mergesort")

    table_cols = [c for c in df.columns if c != "sn_name"]
    table_df = df[table_cols].copy()
    table_df = table_df.round(3)
    columns = [{"name": c, "id": c} for c in table_cols]
    data = table_df.to_dict("records")
    selected = [0] if data else []

    return df.to_json(orient="split"), str(latest), data, columns, selected


@callback(
    Output("sggui-graph", "figure"),
    Output("sggui-last-figure", "data"),
    Input("sggui-table", "selected_rows"),
    Input("sggui-plots", "value"),
    Input("sggui-bin", "value"),
    Input("theme-store", "data"),
    State("sggui-table", "data"),
    State("sggui-results-data", "data"),
    State("sggui-last-figure", "data"),
)
def plot_selected(sel_rows, plot_flags, bin_size, theme, table_data, json_data, stored_fig):
    dark = theme == "dark"

    if not sel_rows or not table_data or not json_data:
        fig, fig_dict = _fallback_figure(stored_fig, dark)
        return fig, fig_dict

    if plot_flags is None:
        plot_flags = _ALL_PLOT_VALUES

    df_all = pd.read_json(json_data, orient="split")
    selected_index = sel_rows[0]
    if selected_index >= len(df_all):
        selected_index = 0
    row = df_all.iloc[selected_index]

    fig = _base_figure(dark)
    y_extents = []

    def _record(values):
        if values is None:
            return
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            y_extents.extend([float(arr.min()), float(arr.max())])

    def _add_trace(key, frame, name):
        nonlocal traces_added
        if key not in plot_flags or frame is None or frame.empty:
            return
        clean = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=clean["wav"],
                y=clean["flux"],
                mode="lines",
                name=name,
                line=_line_style(key, dark),
            )
        )
        _record(clean["flux"].to_numpy(dtype=float))
        traces_added = True

    obs_path = _under_root(RESULTS_DIR, row.get("SPECTRUM"), NGSF_BASE)
    gal_rel = None if pd.isna(row.get("GALAXY")) else str(row.get("GALAXY"))
    gal_path = (
        NGSF_BASE / "bank" / "binnings" / "10A" / "gal" / gal_rel
        if gal_rel else None
    )
    sn_path = _under_root(NGSF_BASE, row.get("sn_name"))

    redshift = _safe_float(row.get("Z"))
    redshift = 0.0 if redshift is None else redshift

    extinction_magnitude = _safe_float(row.get("A_v"))
    extinction_magnitude = 0.0 if extinction_magnitude is None else extinction_magnitude

    const_sn = _safe_float(row.get("CONST_SN"))
    const_sn = 1.0 if const_sn is None else const_sn

    const_gal = _safe_float(row.get("CONST_GAL"))
    const_gal = 1.0 if const_gal is None else const_gal

    bin_step = _safe_float(bin_size)
    if bin_step is not None and bin_step <= 0:
        bin_step = None

    traces_added = False

    obs_plot = None
    if obs_path and obs_path.exists():
        obs_data = read_two_col_txt(obs_path)
        if not obs_data.empty:
            obs_plot = normalize_flux(obs_data)
            if bin_step:
                obs_plot = binspec(
                    obs_plot,
                    obs_plot["wav"].min(),
                    obs_plot["wav"].max(),
                    bin_step,
                )
            obs_plot = obs_plot.replace([np.inf, -np.inf], np.nan).dropna()

    if obs_plot is None or obs_plot.empty:
        fig, fig_dict = _fallback_figure(stored_fig, dark)
        return fig, fig_dict

    plot_wavelength = obs_plot["wav"].to_numpy(dtype=float)
    gal_scaled = None
    if gal_path and gal_path.exists():
        gal_data = read_two_col_txt(gal_path)
        gal_scaled = _fit_grid_component(
            gal_data,
            target_wavelength=plot_wavelength,
            redshift=redshift,
            scale=const_gal,
        )
    sn_template = None
    sn_fitted = None
    if sn_path and sn_path.exists():
        sn_data = read_two_col_txt(sn_path)
        if not sn_data.empty:
            sn_template = sn_data.copy()
            if bin_step:
                sn_template = binspec(
                    sn_template,
                    sn_template["wav"].min(),
                    sn_template["wav"].max(),
                    bin_step,
                )
            sn_fitted = _fit_grid_component(
                sn_data,
                target_wavelength=plot_wavelength,
                redshift=redshift,
                scale=const_sn,
                extinction_magnitude=extinction_magnitude,
            )

    observation_minus_galaxy = None
    if gal_scaled is not None and not gal_scaled.empty:
        observation_minus_galaxy = pd.DataFrame(
            {
                "wav": plot_wavelength,
                "flux": (
                    obs_plot["flux"].to_numpy(dtype=float)
                    - gal_scaled["flux"].to_numpy(dtype=float)
                ),
            }
        )

    _add_trace("obs", obs_plot, "Observation")
    _add_trace("gal", gal_scaled, f"Galaxy ({row.get('GALAXY')})")
    _add_trace("tem", sn_template, "Template")
    _add_trace("ute", sn_fitted, "Normalized Template")
    _add_trace("omg", observation_minus_galaxy, "Observation − Galaxy")

    if not traces_added:
        fig, fig_dict = _fallback_figure(stored_fig, dark)
        return fig, fig_dict

    if y_extents:
        y_min = float(np.min(y_extents))
        y_max = float(np.max(y_extents))
        spread = y_max - y_min
        pad = (max(abs(y_min), 1.0) * 0.05) if spread <= 0 else spread * 0.05
        fig.update_yaxes(range=[y_min - pad, y_max + pad], autorange=False)

    fig_dict = fig.to_dict()
    return fig, fig_dict


@callback(
    Output("sggui-bestfit", "children"),
    Input("sggui-table", "selected_rows"),
    State("sggui-results-data", "data"),
)
def update_bestfit(sel_rows, json_data):
    if not json_data:
        return ""
    df_all = pd.read_json(json_data, orient="split")
    idx = sel_rows[0] if sel_rows else 0
    if idx >= len(df_all):
        idx = 0
    row = df_all.iloc[idx]

    sn_type = row.get("SN", "unknown")
    host = row.get("GALAXY", "unknown")
    z = _fmt_float(row.get("Z"), nd=4)
    av = _fmt_float(row.get("A_v"), nd=2)
    frac_sn = _fmt_pct(row.get("Frac(SN)"))
    chi = _fmt_float(row.get("CHI2/dof"), nd=2)

    def _line(label, value):
        return html.Div(
            [html.Strong(label), html.Span(value)],
            style={"margin": "2px 0"},
        )

    return html.Div(
        [
            _line("Supernova Type: ", sn_type),
            _line("Host Galaxy: ", host),
            _line("Redshift (z): ", z),
            _line("Extinction (A_v): ", av),
            _line("SN Contribution: ", frac_sn),
            _line("Chi-squared: ", chi),
        ]
    )


@callback(
    Output("sggui-bestfit-wrapper", "style"),
    Input("sggui-results-data", "data"),
)
def toggle_bestfit_display(json_data):
    if not json_data:
        return {"display": "none"}
    try:
        df = pd.read_json(json_data, orient="split")
    except Exception:
        return {"display": "none"}
    return {"display": "block"} if not df.empty else {"display": "none"}