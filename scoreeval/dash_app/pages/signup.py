import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import callback
from dash.dependencies import Input, Output, State
from dash_app.database import create_account

dash.register_page(__name__)

# Create username and password form for signing up
username_input = html.Div(
    [
        dbc.Label("Username", html_for="example-user", style={"font-weight": "500"}),
        dbc.Input(placeholder="Enter your username", type="text", id="uname-box-acc", \
                  style={"padding": "10px", "margin-bottom": "20px"}),
    ],
)

password_input = html.Div(
    [
        dbc.Label("Password", html_for="example-password", style={"font-weight": "500"}),
        dbc.Input(placeholder="Enter your password", type="password", id="pwd-box-acc", \
                style={"padding": "10px", "margin-bottom": "20px"})
    ],
)

form = dbc.Form([username_input, password_input])

# Signup screen layout (similar to login screen layout)
layout = html.Div(
    [
        form,
        html.Button("Create Account", n_clicks=0, type="submit", id="sign-up-button", style={
                        "background-color": "#003366",   # dark blue
                        "border-color": "#003366",
                        "color": "white"
        }),
        dbc.Modal(
            [
                dbc.ModalBody("Account created successfully...", id="modal-signup-body"),
                dbc.ModalFooter(
                    html.Button("Close", n_clicks=0, id="close-centered", style={
                        "background-color": "#003366",   # dark blue
                        "border-color": "#003366",
                        "color": "white"
                    })
                ),
            ],
            id="signup-modal-centered",
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

# Callback after signing up
# we should indicate success/failure to the user who
# should then login or retry the signup process
@callback(
    Output("signup-modal-centered", "is_open"),
    Output("modal-signup-body", "children"),
    Output("uname-box-acc", "value"),
    Output("pwd-box-acc", "value"),
    Input("sign-up-button", "n_clicks"),
    Input("close-centered", "n_clicks"),
    State("uname-box-acc", "value"),
    State("pwd-box-acc", "value"),
    State("signup-modal-centered", "is_open"),
    prevent_initial_call=True,
)
def acc_button_click(n_clicks, n_clicks_close, username, password, is_open):
    if is_open:
        return False, "", "", ""

    if n_clicks:
        success = create_account(username, password)
        if success is None:
            return True, "Please type a valid username/password!", username, password
        if success:
            return True, "Account created successfully, login to access your account.", "", ""
        return True, "Username already exists, please login.", "", "" 
    return False, "", "", ""
