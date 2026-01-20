import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import callback
from dash_app.database import SessionLocal, User, auth_func
from dash.dependencies import Input, Output, State
from flask_login import login_user
from dash.exceptions import PreventUpdate

dash.register_page(__name__)

# Create username and password form to collect login details
username_input = html.Div(
    [
        dbc.Label("Username", html_for="example-user", style={"font-weight": "500"}),
        dbc.Input(placeholder="Enter your username", type="text", id="uname-box", \
                  style={"padding": "10px", "margin-bottom": "20px"}),
    ],
)

password_input = html.Div(
    [
        dbc.Label("Password", html_for="example-password", style={"font-weight": "500"}),
        dbc.Input(placeholder="Enter your password", type="password", id="pwd-box", \
                style={"padding": "10px", "margin-bottom": "20px"})
    ],
)

form = dbc.Form([username_input, password_input])

# Login screen
layout = html.Div(
    [
        form,
        html.Button("Login", n_clicks=0, type="submit", id="login-button", style={
                        "background-color": "#003366",   # dark blue
                        "border-color": "#003366",
                        "color": "white"
        }),
        html.Div(children=[
            html.Span("Don't have an account? "),
            dcc.Link(
                "Sign up",
                href="/signup",
                refresh=True,
                style={"color": "#003366", "font-weight": "500"}
        ),
        ], id="output-state", style={"padding-top": "10px"}),

        dbc.Modal(
            [
                dbc.ModalBody("Wrong details! Sign up if you don't have an account.", id="modal-login-body"),
                dbc.ModalFooter(
                    html.Button("Close", n_clicks=0, id="close-login-centered", style={
                        "background-color": "#003366",   # dark blue
                        "border-color": "#003366",
                        "color": "white"
                    })
                ),
            ],
            id="login-modal-centered",
            centered=True,
            is_open=False,
        ),

    ],
    style={
        "display": "block",
        "padding": "50px",
        "margin": "auto",
        "width": "50%"
    },
)

# Callback for logging in into the application
@callback(
    Output("login-modal-centered", "is_open"),
    Output("modal-login-body", "children"),
    Output("init-login", "data"),
    Input("login-button", "n_clicks"),
    Input("close-login-centered", "n_clicks"),
    State("uname-box", "value"),
    State("pwd-box", "value"),
    State("login-modal-centered", "is_open"),
    prevent_initial_call=True,
)
def login_button_click(n_clicks, _, username, password, is_open):
    if is_open:
        return False, "", False
    
    if n_clicks > 0:
        session = SessionLocal()
        try:
            user = session.query(User).filter_by(id=username).first()
            if user and auth_func(username, password):
                login_user(user)  # <-- THIS logs the user in
                return False, "", True
            return True, "Wrong details! Sign up if you don't have an account.", False
        finally:
            session.close()
    else:
        raise PreventUpdate
