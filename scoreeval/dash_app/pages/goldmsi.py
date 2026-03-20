from dash import html, dcc
import dash_bootstrap_components as dbc
import dash
from dash.dependencies import Input, Output, State
from dash import callback
from flask_login import current_user
from dash_app.database import load_goldmsi_quests, add_goldmsi_response
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/goldmsi")  # goldmsi page

# define a function to return question component
def gold_msi_component(quests: list) -> list:
    comps = []

    for q in quests:
        items = q["encoding_string"].split(";")
        opts = []
        if q["scale_type"] != "agreement":      
            for item in items:
                opts.append({"label": item, "value": item})
        else:
            text_map = {'0': "No", '1': "Yes"}
            for item in items:
                opts.append({"label": text_map[item], "value": item})
                opts.sort(key=lambda x: x["value"], reverse=True)

        # Create a centered Dropdown inside a column
        dropdown_row = dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id=f"dropdown_{q["id"]}",
                    options=opts,
                    placeholder="Select your answer...",
                    style={"width": "100%", "align-items": "center"}  # full width of column
                ),
                width=6  # half of row
            ),
            justify="center",
            className="mb-3 g-0"  # spacing below dropdown
        )

        # Wrap question in a Card
        card = dbc.Card(
            [
                html.H3(f"Question {q['id']}"),
                html.P(f"{q['text']}"),
                dropdown_row
            ],
            body=True,
            className="mb-4 shadow-sm"  # spacing between cards + subtle shadow
        )

        # Wrap the card in a row to center it and set width
        card_row = dbc.Row(
            dbc.Col(
                card,
                width=6 # column takes 50% of row
            ),
            justify="center",
            className="mb-4 g-0"  # spacing between questions
        )
        
        comps.append(card_row)

    submit_row = dbc.Row(
        dbc.Col(
            dbc.Button(
                    "Submit Responses",
                    id="submit-gold", style={
                        "background-color": "#003366",   # dark blue
                        "border-color": "#003366",
                        "color": "white"},
                    size="lg"
                ),
            width=6  # button column takes 50% of row
        ),
        justify="center",
        className="mb-5 g-0"  # spacing from last question
    )
    popup = dbc.Modal(
        [
            dbc.ModalBody("WARNING! Answer all questions to proceed...", id="modal-msiwarning-body"),
            dbc.ModalFooter(
                html.Button("Close", n_clicks=0, id="close-msiwarning-centered", style={
                    "background-color": "#003366",   # dark blue
                    "border-color": "#003366",
                    "color": "white"
                })
            ),
        ],
        id="msiwarning-modal-centered",
        centered=True,
        is_open=False,
    )
    comps.append(submit_row)
    comps.append(popup)
    return html.Div(id="dropdown_gold",
        children=comps
    )

layout = html.Div(
    id="gmsi-id",
    style={
        "text-align": "center", 
    },
    children=gold_msi_component(load_goldmsi_quests())
)

# Callback for Modal Warning
@callback(
    Output("msiwarning-modal-centered", "is_open"),
    Output("modal-msiwarning-body", "children"),
    Input("submit-gold", "n_clicks"),
    Input("close-msiwarning-centered", "n_clicks"),
    State("msiwarning-modal-centered", "is_open"),
    State("dropdown_1", "value"),
    State("dropdown_2", "value"),
    State("dropdown_3", "value"),
    State("dropdown_4", "value"),
    State("dropdown_5", "value"),
    State("dropdown_6", "value"),
    State("dropdown_7", "value"),
    State("dropdown_8", "value"),
    State("dropdown_9", "value"),
    State("dropdown_10", "value"),
    State("dropdown_11", "value"),
    State("dropdown_12", "value"),
    State("dropdown_13", "value"),
    State("dropdown_14", "value"),
    State("dropdown_15", "value"),
    State("dropdown_16", "value"),
    prevent_initial_call=True
)
def msi_modal_warning(_, close_modal, is_open, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, \
    d11, d12, d13, d14, d15, d16):
    answers = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, \
               d14, d15, d16]
    
    if is_open:
        return False, ""
    
    for i, ans in enumerate(answers):
        if ans is None:
            return True, "WARNING! Answer all questions before proceeding with the test."
    
    return False, ""


@callback(
    Output("submit-gold-msi", "data"),
    Input("submit-gold", "n_clicks"),
    State("dropdown_1", "value"),
    State("dropdown_2", "value"),
    State("dropdown_3", "value"),
    State("dropdown_4", "value"),
    State("dropdown_5", "value"),
    State("dropdown_6", "value"),
    State("dropdown_7", "value"),
    State("dropdown_8", "value"),
    State("dropdown_9", "value"),
    State("dropdown_10", "value"),
    State("dropdown_11", "value"),
    State("dropdown_12", "value"),
    State("dropdown_13", "value"),
    State("dropdown_14", "value"),
    State("dropdown_15", "value"),
    State("dropdown_16", "value"),
)
def submit_gold(n_clicks, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, \
    d11, d12, d13, d14, d15, d16):
    answers = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, \
               d14, d15, d16]
    response_dict = {}
    
    for i, ans in enumerate(answers):
        if ans is None:
            return False
        response_dict[i+1] = ans
    
    add_goldmsi_response(response_dict, current_user.get_id())
    return True
