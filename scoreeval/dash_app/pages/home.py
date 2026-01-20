from dash import html, dcc
import dash_bootstrap_components as dbc
import dash
from dash.dependencies import Input, Output, State
from dash import callback
from flask_login import current_user

dash.register_page(__name__, path="/")  # home page

layout = html.Div(
    id="home-id",
    style={
        "text-align": "center", 
        "padding": "50px",
    },
    children=[]
)

@callback(
    Output("home-id", "children"),
    Input("home-id", "children"),
)
def load_output(n):
    out_div = html.Div(children=[
        html.H1(id="welcome", children="Welcome!"),
    html.P("Please read the instructions before attempting the questionnaire."),

    dbc.Row([
        dbc.Col([
            dcc.Link(dbc.Button("Read Instructions", id="start-btn", style={
                "background-color": "#003366",   # dark blue
                "border-color": "#003366",
                "color": "white"
            }), href="/instructions", style={"margin": "8px"}),

            dcc.Link(dbc.Button("Proceed with questions", id="start-btn", style={
                "background-color": "#003366",   # dark blue
                "border-color": "#003366",
                "color": "white"
            }), href="/questions", style={"margin": "8px"})
        ], align="center")
    ]),
    ])

    if current_user.is_authenticated:
        return out_div
    return html.Div(children=[])
