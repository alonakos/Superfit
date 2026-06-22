import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server

# ── Navbar — matches reference image exactly ───────────────────
# Brand left | Input/Output tabs center | icons right
navbar = html.Nav(
    className="superfit-navbar d-flex align-items-center justify-content-between",
    children=[
        # LEFT: brand
        html.A(
            className="d-flex align-items-center text-decoration-none",
            href="/sfgui",
            children=[
                html.Span("✦", className="brand-star me-2"),
                html.Span("Superfit", style={
                    "fontWeight": "700",
                    "fontSize": "1.05rem",
                    "color": "var(--text)",
                }),
            ],
        ),

        # CENTER: Input / Output tabs
        html.Div(
            className="d-flex",
            children=dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink(
                        "Input", href="/sfgui", active="exact",
                        className="nav-tab-link",
                    )),
                    dbc.NavItem(dbc.NavLink(
                        "Output", href="/sggui", active="exact",
                        className="nav-tab-link",
                    )),
                ],
                className="nav-tabs superfit-center-tabs",
                style={"border": "none"},
            ),
        ),

        # RIGHT: theme toggle
        html.Button(
            "🌙",
            id="theme-toggle",
            className="theme-toggle-btn",
            title="Toggle dark/light mode",
            n_clicks=0,
        ),
    ],
    style={"position": "sticky", "top": "0", "zIndex": "1000"},
)

# Hidden dcc.Tabs for sync_navigation (original logic)
tabs = dcc.Tabs(
    id="tabs",
    value="tab-sfgui",
    children=[
        dcc.Tab(label="Input",  value="tab-sfgui"),
        dcc.Tab(label="Output", value="tab-sggui"),
    ],
    persistence=True,
    persistence_type="session",
    style={"display": "none"},
)

app.layout = html.Div(
    id="theme-wrapper",
    **{"data-theme": "light"},
    children=[
        dcc.Store(id="theme-store", storage_type="local", data="light"),
        dcc.Store(id="run-flag", storage_type="session"),
        dcc.Location(id="url"),
        html.Div(id="theme-html-sync", style={"display": "none"}),
        navbar,
        tabs,
        dash.page_container,
    ],
)

# ── Propagate theme to <html>/<body> ────────────────────────────────────
# data-theme lives on #theme-wrapper (inside body), so Bootstrap's
# white body background bleeds through unless we mirror it to <html>.
app.clientside_callback(
    """
    function(theme) {
        var t = theme || 'light';
        document.documentElement.setAttribute('data-theme', t);
        document.body.setAttribute('data-theme', t);
        return '';
    }
    """,
    Output("theme-html-sync", "children"),
    Input("theme-store", "data"),
)

# ── Original sync_navigation (preserved exactly) ───────────────
PATH_FOR_TAB = {"tab-sfgui": "/sfgui", "tab-sggui": "/sggui"}
TAB_FOR_PATH = {
    "/":      "tab-sfgui",
    "/sfgui": "tab-sfgui",
    None:     "tab-sfgui",
    "/sggui": "tab-sggui",
}

@app.callback(
    Output("url",  "pathname", allow_duplicate=True),
    Output("tabs", "value",    allow_duplicate=True),
    Input("tabs",  "value"),
    Input("url",   "pathname"),
    prevent_initial_call=True,
)
def sync_navigation(tab_value, pathname):
    ctx = dash.callback_context
    if not ctx.triggered_id:
        raise PreventUpdate
    if ctx.triggered_id == "tabs":
        new_path = PATH_FOR_TAB.get(tab_value)
        if not new_path:
            raise PreventUpdate
        return new_path, dash.no_update
    if ctx.triggered_id == "url":
        new_tab = TAB_FOR_PATH.get(pathname)
        if not new_tab:
            return dash.no_update, dash.no_update
        target_path = PATH_FOR_TAB.get(new_tab)
        if target_path and pathname != target_path:
            return target_path, new_tab
        return dash.no_update, new_tab
    raise PreventUpdate


# ── Single unified theme callback ─────────────────────────────
@app.callback(
    Output("theme-wrapper", "data-theme"),
    Output("theme-toggle",  "children"),
    Output("theme-store",   "data"),
    Input("theme-store",    "data"),
    Input("theme-toggle",   "n_clicks"),
    State("theme-store",    "data"),
    prevent_initial_call=False,
)
def manage_theme(stored_on_load, n_clicks, current_stored):
    ctx = dash.callback_context
    triggered = ctx.triggered_id if ctx.triggered_id else "theme-store"
    if triggered == "theme-toggle" and n_clicks:
        new_theme = "dark" if (current_stored or "light") == "light" else "light"
    else:
        new_theme = stored_on_load or "light"
    icon = "☀️" if new_theme == "dark" else "🌙"
    return new_theme, icon, new_theme


if __name__ == "__main__":
    app.run(debug=True, port=8050)