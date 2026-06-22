from GUISuperfit.mergedapp.app import app as dash_app, server

app = server


if __name__ == "__main__":
    dash_app.run_server(host="0.0.0.0", port=8050, debug=False)