import os
import json
import base64
import io
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd
import plotly.graph_objects as go


dash.register_page(__name__, path="/sfgui")

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_NGSF_DIR = PROJECT_ROOT / "NGSF"

BASE_DIR = Path(os.environ.get("NGSF_DIR", str(DEFAULT_NGSF_DIR))).resolve()
UPLOAD_DIR = Path(os.environ.get("NGSF_UPLOAD_DIR", str(BASE_DIR))).resolve()
RESULTS_DIR = Path(os.environ.get("NGSF_RESULTS_DIR", str(BASE_DIR / "results"))).resolve()

LOWER_LAM = 3000
UPPER_LAM = 10000
DEFAULT_EPOCH_RANGE = [-30, 171]

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {}


# Progress helpers
def _write_progress(path, percent, message=""):
    payload = {
        "percent": int(max(0, min(100, percent))),
        "message": message,
        "ts": time.time(),
    }
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _read_progress(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return int(data.get("percent", 0)), data.get("message", "")
    except Exception:
        return 0, ""


# SN categories
sn_categories = {
    "IA": ["Ia 02es-like", "Ia-02cx like", "Ia-CSM-(ambigious)", "Ia 91T-like", "Ia-CSM",
           "Ia-norm", "Ia 91bg-like", "Ia-rapid"],
    "IB": ["Ib", "Ca-Ib"],
    "II": ["IIb-flash", "II", "IIb", "II-flash", "ILRT"],
    "SLSN": ["SLSN-II", "SLSN-IIn", "SLSN-I", "SLSN-Ib", "SLSN-IIb"],
    "Other": [
        "computed", "TDE He", "Ca-Ia", "super_chandra", "IIn", "FBOT", "Ibn", "TDE H",
        "SN - Imposter", "TDE H+He", "Ic", "Ia-pec", "Ic-BL", "Ic-pec",
    ],
}

_category_names = list(sn_categories.keys())
_CATEGORY_COUNT = len(_category_names)

_all_subtypes = [s for subs in sn_categories.values() for s in subs]
_sn_subtype_options = [{"label": s, "value": s} for s in _all_subtypes]
_default_subtypes = ["Ia-norm", "Ia 91bg-like"]
_default_cats = ["IA"]

_DISABLED_CHECKLIST_STYLE = {"pointerEvents": "none", "opacity": 0.5}
_ENABLED_CHECKLIST_STYLE  = {"pointerEvents": "auto", "opacity": 1.0}


def _sn_state(disabled: bool):
    style = _DISABLED_CHECKLIST_STYLE if disabled else _ENABLED_CHECKLIST_STYLE
    return [disabled] * _CATEGORY_COUNT, [style.copy() for _ in range(_CATEGORY_COUNT)]


galaxy_options = [
    {"label": g, "value": g}
    for g in ["E", "S0", "Sa", "Sb", "Sc", "SB1", "SB2", "SB3", "SB4", "SB5", "SB6"]
]

# Smooth resampler
def _smooth_resample(df: pd.DataFrame, lo: float, hi: float, step: float = 5.0) -> pd.DataFrame:
    if df.empty:
        return df
    wav = df["wavelength"].to_numpy(dtype=float)
    flx = df["flux"].to_numpy(dtype=float)

    actual_lo = max(lo, wav.min())
    actual_hi = min(hi, wav.max())
    if actual_lo >= actual_hi:
        return df

    data_range = actual_hi - actual_lo
    n_points   = len(wav[(wav >= actual_lo) & (wav <= actual_hi)])

    if n_points > 0:
        native_step = data_range / n_points
        adaptive_step = max(native_step, data_range / 500)
    else:
        adaptive_step = step

    grid = np.arange(actual_lo, actual_hi, adaptive_step)
    if len(grid) < 2:
        return df

    flx_interp = np.interp(grid, wav, flx)
    out = pd.DataFrame({"wavelength": grid, "flux": flx_interp})

    # Adaptive window: ~75Å worth of points, min 3
    window = max(3, int(75 / adaptive_step))
    out["flux"] = out["flux"].rolling(window, center=True, min_periods=1).mean()
    return out


# Layout components
known_redshift_tab = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                dbc.Col(html.Label("z"), width="auto"),
                dbc.Col(
                    dbc.Input(
                        id="sfgui-z-known",
                        type="number",
                        size="sm",
                        persistence=True,
                        persistence_type="session",
                    ),
                    width=5,
                ),
            ],
            className="align-items-center",
        )
    ),
    className="mt-1",
)

redshift_range_tab = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                dbc.Col(html.Label("z1"), width="auto"),
                dbc.Col(
                    dbc.Input(
                        id="sfgui-z1",
                        type="number",
                        size="sm",
                        persistence=True,
                        persistence_type="session",
                    ),
                    width=2,
                ),
                dbc.Col(html.Label("z2"), width="auto"),
                dbc.Col(
                    dbc.Input(
                        id="sfgui-z2",
                        type="number",
                        size="sm",
                        persistence=True,
                        persistence_type="session",
                    ),
                    width=2,
                ),
                dbc.Col(html.Label("dz"), width="auto"),
                dbc.Col(
                    dbc.Input(
                        id="sfgui-dz",
                        type="number",
                        size="sm",
                        persistence=True,
                        persistence_type="session",
                    ),
                    width=2,
                ),
            ],
            className="align-items-center",
        )
    ),
    className="mt-1",
)

z_tabs = dbc.Tabs(
    [
        dbc.Tab(known_redshift_tab, label="Known redshift"),
        dbc.Tab(redshift_range_tab, label="Redshift range"),
    ]
)

uploader = html.Div(
    [
        dcc.Upload(
            id="sfgui-upload",
            children=html.Div([
                html.I(className="ti ti-upload me-2"),
                "Drag & drop or ", html.A("select spectrum .dat"),
                html.Br(),
                html.Small("Supports .dat, .txt", className="text-muted"),
            ]),
            className="sf-upload",
            accept=".dat,.txt",
            multiple=False,
        ),
        html.Small(id="sfgui-upload-status"),
    ]
)


def _sn_accordion():
    content = [
        html.Div(
            [
                html.Div(
                    dbc.Checkbox(
                        id={"type": "sn-cat-toggle", "category": category},
                        value=(category in _default_cats),
                        label=html.Strong(category),
                        persistence=True,
                        persistence_type="session",
                        disabled=True,
                    ),
                    style={"marginBottom": "5px"},
                ),
                dcc.Checklist(
                    id={"type": "sn-subtypes", "category": category},
                    options=[{"label": s, "value": s} for s in subs],
                    value=(
                        subs
                        if category in _default_cats
                        else [s for s in subs if s in _default_subtypes]
                    ),
                    persistence=True,
                    persistence_type="session",
                    inputStyle={"marginRight": "6px"},
                    labelStyle={
                        "display": "inline-block",
                        "marginRight": "18px",
                        "marginBottom": "6px",
                    },
                    style=_DISABLED_CHECKLIST_STYLE.copy(),
                ),
                html.Hr(),
            ],
            style={"marginBottom": "10px"},
        )
        for category, subs in sn_categories.items()
    ]

    hidden_bridge = dcc.Checklist(
        id="sfgui-sn-types",
        options=_sn_subtype_options,
        value=_default_subtypes,
        style={"display": "none"},
        persistence=True,
        persistence_type="session",
    )

    toggle_row = html.Div(
        [
            html.Label("Supernova types", style={"fontWeight": "bold"}, className="mb-0"),
            dbc.Button(
                "Select All",
                id="sfgui-sn-select",
                size="sm",
                color="link",
                className="p-0",
                disabled=True,
            ),
        ],
        className="d-flex justify-content-between align-items-center mb-2",
    )

    return html.Div([toggle_row, html.Div(content), hidden_bridge, html.Hr()])


sn_checklist = dbc.Card(
    dbc.CardBody(
        [
            _sn_accordion(),
            html.Label("Epoch Range Slider"),
            dcc.RangeSlider(
                id="sfgui-epoch-range",
                min=-100,
                max=700,
                step=5,
                value=DEFAULT_EPOCH_RANGE,
                marks={i: str(i) for i in range(-100, 701, 100)},
                allowCross=False,
                pushable=5,
                tooltip={"placement": "bottom", "always_visible": False},
                persistence=True,
                persistence_type="session",
            ),
            html.Small(id="sfgui-epoch-label"),
            html.Hr(),
            html.Div(
                [
                    html.Label(
                        "Galaxies",
                        style={"fontWeight": "bold"},
                        className="mb-0",
                    ),
                    dbc.Button(
                        "Select All",
                        id="sfgui-gal-select",
                        size="sm",
                        color="link",
                        className="p-0",
                    ),
                ],
                className="d-flex justify-content-between align-items-center mb-2",
            ),
            dbc.Checklist(
                id="sfgui-galaxies",
                options=galaxy_options,
                value=["E", "S0", "Sa", "Sb", "Sc"],
                inline=True,
                persistence=True,
                persistence_type="session",
            ),
            html.Hr(),
            html.Label("Reddening", style={"fontWeight": "bold"}, className="mb-2 d-block"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("A_hi"),
                                dbc.Input(
                                    id="sfgui-a-hi",
                                    type="number",
                                    value=3.0,
                                    step=0.1,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                            ],
                            size="sm",
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("A_lo"),
                                dbc.Input(
                                    id="sfgui-a-lo",
                                    type="number",
                                    value=0.0,
                                    step=0.1,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                            ],
                            size="sm",
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("A_i"),
                                dbc.Input(
                                    id="sfgui-a-int",
                                    type="number",
                                    value=0.1,
                                    step=0.1,
                                    min=0.0001,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                            ],
                            size="sm",
                        ),
                        width=4,
                    ),
                ],
                className="mb-2",
            ),
        ]
    ),
    className="mt-2",
)

spectrum_graph = dcc.Graph(
    id="sfgui-graph",
    config={"displayModeBar": False, "responsive": True},
    style={"height": "60vh"},
)

wavelength_slider = html.Div(
    [
        html.Label("Wavelength Range Slider"),
        dcc.RangeSlider(
            id="sfgui-wave-range",
            min=LOWER_LAM,
            max=UPPER_LAM,
            value=[LOWER_LAM, UPPER_LAM],
            step=10,
            marks={i: str(i) for i in range(LOWER_LAM, UPPER_LAM + 1, 1000)},
            allowCross=False,
            updatemode="mouseup",
            tooltip={"placement": "bottom", "always_visible": False},
            persistence=True,
            persistence_type="session",
        ),
        html.Small(id="sfgui-wave-label"),
    ],
    className="mt-2",
)

btn_generate = dbc.Button(
    "Generate JSON",
    color="secondary",
    id="sfgui-generate",
    className="me-2",
    disabled=True,
)
btn_run = dbc.Button("Run Fit", color="primary", id="sfgui-run", disabled=True, className="me-2")
run_progress_row = html.Div(
    id="sfgui-progress-row",
    style={"display": "none"},
    className="mt-2",
    children=[
        # Status text on its own line above the bar
        html.Small(
            id="sfgui-run-status",
            className="text-muted d-block mb-1",
        ),
        # Full-width bar — percentage rendered as the bar label inside it
        dbc.Progress(
            id="sfgui-progress",
            value=0,
            label="",
            striped=True,
            animated=False,
            style={"height": "1.6rem", "width": "100%", "fontSize": ".85rem"},
        ),
    ],
)
run_timer = dcc.Interval(
    id="sfgui-run-timer",
    interval=1000,
    disabled=True,
    n_intervals=0,
)
run_state = dcc.Store(id="sfgui-run-state", storage_type="session")
redirect = dcc.Location(id="sfgui-redirect", refresh=True)
btn_clear = dbc.Button("Clear", color="danger", id="sfgui-clear", className="mr-1")

json_status = html.Div(id="sfgui-json-status", className="text-muted small mb-2")
json_output = html.Pre(
    id="sfgui-json",
    style={"whiteSpace": "pre-wrap", "maxHeight": "35vh", "overflowY": "auto"},
)
run_status_placeholder = None  # sfgui-run-status now lives inside sfgui-progress-row
download_json = dcc.Download(id="sfgui-download")

store_filename = dcc.Store(id="sfgui-store-fn", storage_type="session")
store_df = dcc.Store(id="sfgui-store-df", storage_type="session")
store_wave_bounds = dcc.Store(id="sfgui-store-wave", storage_type="session")

layout = html.Div(
    [
        store_filename,
        store_df,
        store_wave_bounds,
        run_state,
        run_timer,
        redirect,
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                uploader,
                                z_tabs,
                                sn_checklist,
                                html.Div(
                                    [
                                        btn_run,
                                        btn_generate,
                                        btn_clear,
                                        download_json,
                                    ],
                                    className="mt-2",
                                ),
                                run_progress_row,
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Parameters JSON"),
                                        dbc.CardBody([json_status, json_output]),
                                    ],
                                    className="mt-3",
                                ),
                            ],
                            md=5,
                        ),
                        dbc.Col(
                            dbc.Card(dbc.CardBody([spectrum_graph, wavelength_slider])),
                            md=7,
                        ),
                    ],
                    className="mt-3",
                ),
            ],
            fluid=True,
        ),
    ]
)


def _build_params(
    filename,
    z_known,
    z1,
    z2,
    dz,
    sn_types,
    epoch_range,
    galaxies,
    a_hi,
    a_lo,
    a_int,
    wave_range=None,
    progress_path=None,
):
    if wave_range and len(wave_range) == 2:
        lower_lam = float(wave_range[0])
        upper_lam = float(wave_range[1])
    else:
        lower_lam = LOWER_LAM
        upper_lam = UPPER_LAM

    object_path = str(UPLOAD_DIR / filename) if filename else "spectrum.dat"

    return {
        "object_to_fit": object_path,
        "use_exact_z": 1 if z_known is not None else 0,
        "z_exact": float(z_known) if z_known is not None else 0.05,
        "z_range_begin": float(z1) if z1 is not None else 0.0,
        "z_range_end": float(z2) if z2 is not None else 0.1,
        "z_int": float(dz) if dz is not None else 0.01,
        "resolution": 10,
        "lower_lam": lower_lam,
        "upper_lam": upper_lam,
        "saving_results_path": f"{RESULTS_DIR}{os.sep}",
        "pkg_dir": str(BASE_DIR),
        "temp_gal_tr": galaxies or ["E", "S0", "Sa", "Sb", "Sc"],
        "temp_sn_tr": sn_types or ["Ia", "Ib", "Ic", "II"],
        "mask_galaxy_lines": False,
        "mask_telluric": False,
        "error_spectrum": "sg",
        "minimum_overlap": 0.5,
        "epoch_low": int(epoch_range[0]) if epoch_range else -20,
        "epoch_high": int(epoch_range[1]) if epoch_range else 300,
        "Alam_low": float(a_lo) if a_lo is not None else 0.0,
        "Alam_high": float(a_hi) if a_hi is not None else 3.0,
        "Alam_interval": float(a_int) if a_int is not None else 0.1,
        "show_plot": False,
        "show_plot_png": True,
        "how_many_plots": 5,
        "progress_path": progress_path,
    }


def _parse_dat(contents, filename):
    _, content_string = contents.split(",", 1)
    data = base64.b64decode(content_string)
    text = data.decode("utf-8", errors="ignore")

    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s[0].isalpha() or s[0] in "#%@":
            continue

        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue

        try:
            wav = float(parts[0])
            flux = float(parts[1])
        except ValueError:
            continue

        rows.append((wav, flux))

    df = pd.DataFrame(rows, columns=["wavelength", "flux"])

    if not df.empty:
        df = (
            df.replace([float("inf"), float("-inf")], pd.NA)
            .dropna()
            .sort_values("wavelength")
            .groupby("wavelength", as_index=False)["flux"]
            .mean()
            .reset_index(drop=True)
        )

    dest = UPLOAD_DIR / filename
    with open(dest, "wb") as f:
        f.write(data)

    return df


# ══════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════

@callback(
    Output("sfgui-upload-status", "children"),
    Output("sfgui-store-fn", "data"),
    Output("sfgui-store-df", "data"),
    Output("sfgui-store-wave", "data"),
    Output("sfgui-generate", "disabled"),
    Output("sfgui-run", "disabled"),
    Output({"type": "sn-cat-toggle", "category": ALL}, "disabled"),
    Output({"type": "sn-subtypes", "category": ALL}, "style"),
    Output("sfgui-sn-select", "disabled"),
    Input("sfgui-upload", "contents"),
    State("sfgui-upload", "filename"),
    prevent_initial_call=True,
)
def upload_file(contents, filename):
    if contents is None:
        raise PreventUpdate

    disabled_list, disabled_styles = _sn_state(disabled=True)

    if not filename:
        return "No file selected", None, None, None, True, True, disabled_list, disabled_styles, True

    df = _parse_dat(contents, filename)
    if df.empty:
        return "File parsed as empty", None, None, None, True, True, disabled_list, disabled_styles, True

    wmin = float(df["wavelength"].min())
    wmax = float(df["wavelength"].max())
    status = html.Span(
        [
            dbc.Badge("Spectrum uploaded", color="success", className="me-2"),
            html.Span(filename, className="fw-semibold"),
        ]
    )
    enabled_list, enabled_styles = _sn_state(disabled=False)

    return (
        status,
        filename,
        df.to_json(orient="split", double_precision=15),
        {"min": wmin, "max": wmax},
        False,
        False,
        enabled_list,
        enabled_styles,
        False,
    )


@callback(
    Output("sfgui-wave-range", "min"),
    Output("sfgui-wave-range", "max"),
    Output("sfgui-wave-range", "value"),
    Output("sfgui-wave-range", "marks"),
    Input("sfgui-store-wave", "data"),
    prevent_initial_call=True,
)
def init_wave_slider(bounds):
    if not bounds:
        raise PreventUpdate
    try:
        mn = int(round(float(bounds["min"])))
        mx = int(round(float(bounds["max"])))
    except (KeyError, TypeError, ValueError):
        mn, mx = LOWER_LAM, UPPER_LAM
    if mn >= mx:
        mn, mx = LOWER_LAM, UPPER_LAM
    val = [mn, mx]
    step = max(100, (mx - mn) // 8)
    marks = {i: str(i) for i in range(mn, mx + 1, step)}
    return mn, mx, val, marks


@callback(
    Output("sfgui-wave-label", "children"),
    Input("sfgui-wave-range", "value"),
)
def wave_label(v):
    if not v:
        raise PreventUpdate
    return f"Wavelength range [{v[0]}, {v[1]}]"


@callback(
    Output("sfgui-epoch-label", "children"),
    Input("sfgui-epoch-range", "value"),
)
def epoch_label(v):
    if not v:
        raise PreventUpdate
    return f"Epoch range [{v[0]}, {v[1]}]"


@callback(
    Output("sfgui-graph", "figure"),
    Input("sfgui-store-df", "data"),
    Input("sfgui-wave-range", "value"),
    Input("theme-store", "data"),
    State("sfgui-store-fn", "data"),
)
def update_graph(df_json, wave_range, theme, filename):
    dark = (theme == "dark")

    if dark:
        bg, paper = "#1a1d27", "#1a1d27"
        grid, tc  = "#2e3347", "#e2e8f0"
        line_col  = "#4cc9f0"
        leg_bg    = "rgba(26,29,39,0.85)"
    else:
        bg, paper = "white", "white"
        grid, tc  = "rgba(0,0,0,0.1)", "#111111"
        line_col  = "#4361ee"
        leg_bg    = "rgba(255,255,255,0.75)"

    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper,
        plot_bgcolor=bg,
        xaxis=dict(title="Wavelength (Å)", tickformat=".0f",
                   gridcolor=grid, color=tc),
        yaxis=dict(title="Flux (erg/s/cm²/Å)",
                   gridcolor=grid, color=tc),
        margin=dict(l=30, r=20, t=20, b=40),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor=leg_bg,
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=11, color=tc),
        ),
        uirevision="spectrum",
    )

    if not df_json:
        return fig

    df = pd.read_json(df_json, orient="split")

    if wave_range and len(wave_range) == 2:
        lo, hi = float(wave_range[0]), float(wave_range[1])
        fig.update_xaxes(range=[lo, hi])
    else:
        lo, hi = LOWER_LAM, UPPER_LAM

    df_plot = _smooth_resample(df, lo, hi, step=5.0)
    if df_plot.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=df_plot["wavelength"],
            y=df_plot["flux"],
            mode="lines",
            name=filename or "spectrum",
            line=dict(width=2, color=line_col),
        )
    )

    return fig


@callback(
    Output("sfgui-json", "children"),
    Output("sfgui-json-status", "children"),
    Output("sfgui-download", "data"),
    Input("sfgui-generate", "n_clicks"),
    State("sfgui-z-known", "value"),
    State("sfgui-z1", "value"),
    State("sfgui-z2", "value"),
    State("sfgui-dz", "value"),
    State("sfgui-sn-types", "value"),
    State("sfgui-epoch-range", "value"),
    State("sfgui-galaxies", "value"),
    State("sfgui-a-hi", "value"),
    State("sfgui-a-lo", "value"),
    State("sfgui-a-int", "value"),
    State("sfgui-wave-range", "value"),
    State("sfgui-store-fn", "data"),
    prevent_initial_call=True,
)
def generate_json(
    n,
    z_known,
    z1,
    z2,
    dz,
    sn_types,
    epoch_range,
    galaxies,
    a_hi,
    a_lo,
    a_int,
    wave_range,
    filename,
):
    if not n:
        raise PreventUpdate

    params = _build_params(
        filename,
        z_known,
        z1,
        z2,
        dz,
        sn_types,
        epoch_range,
        galaxies,
        a_hi,
        a_lo,
        a_int,
        wave_range=wave_range,
    )
    text = json.dumps(params, indent=4)

    base_name = Path(filename).stem if filename else "spectrum"
    safe_base = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in base_name
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_name = f"parameters_{safe_base}_{timestamp}.json"

    with open(RESULTS_DIR / json_name, "w") as f:
        f.write(text)

    status = html.Span(
        [
            dbc.Badge("Parameters saved", color="info", className="me-2"),
            html.Span(json_name, className="text-muted"),
        ]
    )
    return text, status, dict(content=text, filename=json_name)


@callback(
    Output("sfgui-run-state",     "data"),
    Output("sfgui-run-timer",     "disabled"),
    Output("sfgui-progress",      "value"),
    Output("sfgui-progress",      "label"),
    Output("sfgui-progress",      "animated"),
    Output("sfgui-progress-row",  "style"),
    Output("sfgui-run-status",    "children"),
    Output("sfgui-run",           "disabled", allow_duplicate=True),
    Input("sfgui-run", "n_clicks"),
    State("sfgui-z-known", "value"),
    State("sfgui-z1", "value"),
    State("sfgui-z2", "value"),
    State("sfgui-dz", "value"),
    State("sfgui-sn-types", "value"),
    State("sfgui-epoch-range", "value"),
    State("sfgui-galaxies", "value"),
    State("sfgui-a-hi", "value"),
    State("sfgui-a-lo", "value"),
    State("sfgui-a-int", "value"),
    State("sfgui-wave-range", "value"),
    State("sfgui-store-fn", "data"),
    prevent_initial_call=True,
)
def start_fit(n, z_known, z1, z2, dz, sn_types, epoch_range,
              galaxies, a_hi, a_lo, a_int, wave_range, filename):
    if not n:
        raise PreventUpdate

    run_id        = str(uuid.uuid4())
    progress_path = RESULTS_DIR / f"progress_{run_id}.json"
    stdout_path   = RESULTS_DIR / f"run_{run_id}.log"
    stderr_path   = RESULTS_DIR / f"run_{run_id}.err"

    _write_progress(progress_path, 0, "Starting fit")

    params = _build_params(
        filename, z_known, z1, z2, dz, sn_types, epoch_range,
        galaxies, a_hi, a_lo, a_int,
        wave_range=wave_range,
        progress_path=str(progress_path),
    )

    _row_show = {"display": "block"}

    try:
        stdout_f = open(stdout_path, "w")
        stderr_f = open(stderr_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "run.py", json.dumps(params)],
            cwd=str(BASE_DIR), stdout=stdout_f, stderr=stderr_f, text=True,
        )
    except Exception as e:
        return (None, True, 0, "0%", False, _row_show, f"Error: {e}", False)

    RUNS[run_id] = {
        "proc": proc, "stdout": stdout_f, "stderr": stderr_f,
        "progress_path": str(progress_path),
        "stderr_path":   str(stderr_path),
        "started":       time.time(),
    }

    return (
        {"run_id": run_id, "progress_path": str(progress_path),
         "stderr_path": str(stderr_path)},
        False,
        0, "0%", True,
        _row_show,
        "Starting fit…",
        True,
    )


@callback(
    Output("sfgui-progress",     "value",    allow_duplicate=True),
    Output("sfgui-progress",     "label",    allow_duplicate=True),
    Output("sfgui-progress",     "animated", allow_duplicate=True),
    Output("sfgui-progress-row", "style",    allow_duplicate=True),
    Output("sfgui-run-status",   "children", allow_duplicate=True),
    Output("sfgui-run-timer",    "disabled", allow_duplicate=True),
    Output("sfgui-run",          "disabled", allow_duplicate=True),
    Output("run-flag",           "data",     allow_duplicate=True),
    Output("sfgui-redirect",     "pathname"),
    Input("sfgui-run-timer",     "n_intervals"),
    State("sfgui-run-state",     "data"),
    prevent_initial_call=True,
)
def poll_fit(n_intervals, state):
    if not state:
        raise PreventUpdate

    import re as _re
    _row_show = {"display": "block"}
    _row_hide = {"display": "none"}

    run_id           = state.get("run_id")
    run              = RUNS.get(run_id)
    percent, message = _read_progress(state.get("progress_path"))
    label            = f"{percent}%"

    # Filter "Fitting grid point X/Y" → show clean message instead
    if message and _re.search(r"[Ff]itting grid point\s+\d+/\d+", message):
        display_msg = "Fitting…"
    else:
        display_msg = message or "Running…"

    if not run:
        return (percent, label, False, _row_show, f"⚠ {display_msg}",
                True, False, dash.no_update, dash.no_update)

    proc = run["proc"]

    if proc.poll() is None:
        # Still running — animated bar, status text on left
        return (percent, label, True, _row_show, display_msg,
                False, True, dash.no_update, dash.no_update)

    # Process finished
    for key in ("stdout", "stderr"):
        try: run[key].close()
        except Exception: pass
    RUNS.pop(run_id, None)

    if proc.returncode == 0:
        _write_progress(state["progress_path"], 100, "Fit completed")
        return (100, "100%", False, _row_show, "✓ Fit complete — redirecting…",
                True, False,
                {"action": "run", "ts": time.time()},
                "/sggui")

    try:    err = Path(state.get("stderr_path", "")).read_text()
    except Exception: err = ""

    return (percent, label, False, _row_show, f"✗ Fit failed: {err[:120]}",
            True, False, None, dash.no_update)


@callback(
    Output("sfgui-z-known", "value"),
    Output("sfgui-z1", "value"),
    Output("sfgui-z2", "value"),
    Output("sfgui-dz", "value"),
    Output("sfgui-sn-types", "value", allow_duplicate=True),
    Output({"type": "sn-cat-toggle", "category": ALL}, "value", allow_duplicate=True),
    Output({"type": "sn-subtypes", "category": ALL}, "value", allow_duplicate=True),
    Output("sfgui-epoch-range", "value", allow_duplicate=True),
    Output("sfgui-galaxies", "value", allow_duplicate=True),
    Output("sfgui-gal-select", "children", allow_duplicate=True),
    Output("sfgui-a-hi", "value", allow_duplicate=True),
    Output("sfgui-a-lo", "value", allow_duplicate=True),
    Output("sfgui-a-int", "value", allow_duplicate=True),
    Output("sfgui-store-fn", "data", allow_duplicate=True),
    Output("sfgui-store-df", "data", allow_duplicate=True),
    Output("sfgui-store-wave", "data", allow_duplicate=True),
    Output("sfgui-upload-status", "children", allow_duplicate=True),
    Output("sfgui-json", "children", allow_duplicate=True),
    Output("sfgui-json-status", "children", allow_duplicate=True),
    Output("sfgui-run-status", "children", allow_duplicate=True),
    Output("sfgui-sn-select", "children", allow_duplicate=True),
    Output({"type": "sn-cat-toggle", "category": ALL}, "disabled", allow_duplicate=True),
    Output({"type": "sn-subtypes", "category": ALL}, "style", allow_duplicate=True),
    Output("sfgui-sn-select", "disabled", allow_duplicate=True),
    Output("sfgui-generate", "disabled", allow_duplicate=True),
    Output("sfgui-run", "disabled", allow_duplicate=True),
    Output("sfgui-wave-range", "value", allow_duplicate=True),
    Output("run-flag", "data", allow_duplicate=True),
    Output("sfgui-run-state", "data", allow_duplicate=True),
    Output("sfgui-run-timer", "disabled", allow_duplicate=True),
    Output("sfgui-progress", "value",         allow_duplicate=True),
    Output("sfgui-progress", "label",         allow_duplicate=True),
    Output("sfgui-progress-row", "style",     allow_duplicate=True),
    Output("sfgui-redirect", "pathname",      allow_duplicate=True),
    Input("sfgui-clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all(n):
    if not n:
        raise PreventUpdate

    disabled_list, disabled_styles = _sn_state(disabled=True)
    count = _CATEGORY_COUNT
    empty_cats = [False] * count
    empty_subs = [[] for _ in range(count)]

    return (
        None,
        None,
        None,
        None,
        [],
        empty_cats,
        empty_subs,
        DEFAULT_EPOCH_RANGE,
        [],
        "Select All",
        3.0, 0.0, 0.1,
        None, None, None,
        "",
        "", "", "",
        "Select All",
        disabled_list,
        disabled_styles,
        True,
        True,
        True,
        [LOWER_LAM, UPPER_LAM],
        {"action": "clear", "ts": time.time()},
        None,
        True,
        0,
        "",
        {"display": "none"},
        dash.no_update,
    )


@callback(
    Output("sfgui-galaxies", "value"),
    Output("sfgui-gal-select", "children"),
    Input("sfgui-gal-select", "n_clicks"),
    State("sfgui-galaxies", "value")
)
def toggle_galaxies(n, current):
    if not n:
        raise PreventUpdate

    all_gals = [g["value"] for g in galaxy_options]

    if set(current) == set(all_gals):
        return [], "Select All"
    else:
        return all_gals, "Deselect All"


@callback(
    Output({"type": "sn-cat-toggle", "category": ALL}, "value", allow_duplicate=True),
    Output({"type": "sn-subtypes", "category": ALL}, "value", allow_duplicate=True),
    Output("sfgui-sn-types", "value", allow_duplicate=True),
    Output("sfgui-sn-select", "children", allow_duplicate=True),
    Input({"type": "sn-cat-toggle", "category": ALL}, "value"),
    Input("sfgui-sn-select", "n_clicks"),
    Input({"type": "sn-subtypes", "category": ALL}, "value"),
    State({"type": "sn-subtypes", "category": ALL}, "options"),
    prevent_initial_call=True,
)
def _sync_sn_selections(cat_states, select_clicks, sub_values, options_list):
    ctx = dash.callback_context
    if not ctx.triggered_id:
        raise PreventUpdate

    options_list = list(options_list or [])
    count = len(options_list)
    cat_states = list(cat_states or [False] * count)

    if not sub_values:
        sub_values = [[] for _ in range(count)]
    else:
        sub_values = [list(v or []) for v in sub_values]
        if len(sub_values) < count:
            sub_values.extend([[] for _ in range(count - len(sub_values))])
        elif len(sub_values) > count:
            sub_values = sub_values[:count]

    def _aggregate(sub_lists):
        return sorted({s for group in (sub_lists or []) for s in (group or [])})

    triggered_id = ctx.triggered_id

    if triggered_id == "sfgui-sn-select":
        if not count:
            raise PreventUpdate

        all_selected = True
        for opts, vals in zip(options_list, sub_values):
            option_values = [o["value"] for o in (opts or [])]
            if not option_values:
                continue
            if len(vals) != len(option_values) or set(vals) != set(option_values):
                all_selected = False
                break

        select_all = not all_selected
        new_cat_values = [select_all and bool(opts) for opts in options_list]
        new_sub_values = [
            [o["value"] for o in (opts or [])] if select_all else []
            for opts in options_list
        ]
        combined = _aggregate(new_sub_values)
        button_label = "Deselect All" if len(combined) == len(_all_subtypes) else "Select All"
        return new_cat_values, new_sub_values, combined, button_label

    if isinstance(triggered_id, dict):
        trig_type = triggered_id.get("type")

        if trig_type == "sn-cat-toggle":
            category = triggered_id.get("category")
            if category not in _category_names:
                raise PreventUpdate
            idx = _category_names.index(category)
            if idx >= count:
                raise PreventUpdate

            target_options = [o["value"] for o in (options_list[idx] or [])]
            is_active = bool(cat_states[idx])
            sub_values[idx] = target_options if is_active else []
            combined = _aggregate(sub_values)
            button_label = "Deselect All" if len(combined) == len(_all_subtypes) else "Select All"

            normalized_cat = []
            for i, opts in enumerate(options_list):
                has_all = bool(opts) and set(sub_values[i]) == {
                    o["value"] for o in (opts or [])
                }
                normalized_cat.append(has_all)

            return normalized_cat, sub_values, combined, button_label

        if trig_type == "sn-subtypes":
            normalized_cat = []
            for opts, vals in zip(options_list, sub_values):
                option_values = {o["value"] for o in (opts or [])}
                normalized_cat.append(
                    bool(option_values) and set(vals or []) == option_values
                )

            combined = _aggregate(sub_values)
            button_label = "Deselect All" if len(combined) == len(_all_subtypes) else "Select All"

            if normalized_cat == cat_states:
                cat_output = [dash.no_update] * count
            else:
                cat_output = normalized_cat

            return cat_output, sub_values, combined, button_label

    raise PreventUpdate