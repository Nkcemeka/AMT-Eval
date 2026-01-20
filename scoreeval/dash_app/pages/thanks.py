from dash import html
import dash

dash.register_page(__name__, path="/thanks")  # home page

layout = html.Div(
    style={"text-align": "center", "padding": "50px"},
    children=[
        html.H1("Thanks!"),
        html.P("You have answered all questions. Thank you for your participation."),
        html.Br(),
    ]
)
