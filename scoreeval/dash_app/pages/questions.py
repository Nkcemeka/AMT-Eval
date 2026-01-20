from dash import html, dcc, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_app.components import audio_component, nav_btns_component,\
      gen_comp, midi_comp_layout, score_comp_layout, \
      NUM_DIM_MIDI, NUM_DIM_SCORE, audio_trans_comp
import dash
import json
from dash_app.database import add_response
from flask_login import current_user
from dash.exceptions import PreventUpdate

# Register this as a page
dash.register_page(__name__, path="/questions")

# Define layout for question page
layout = html.Div(
    id='main-content',
    children=[
        # Progress bar
        dcc.Store(id="max-btn-timestamp", data=-1000000000),
        dcc.Store(id="last-click-back", data=-1000000000),
        dcc.Store(id="last-click-next", data=-1000000000),
        dcc.Store(id='q-type', data=''),
        dcc.Store(id="dimension-status", data=False),
        dcc.Store(id='reload', data=0),
        dcc.Store(id="current-question", data=-1, storage_type="session"),
        dcc.Store(id="prev-question", data=-2), # No session storage to handle reloads effectively
        
        html.Div(
                children=[dbc.Progress(id="progress-bar", color="success", className="mb-3", \
                             value=0, striped=True, animated=True, \
                            style={"height": "30px", "width": "100%"}),
                        ],
                style={"width": "50%", "text-align": "center", "margin": "auto", "padding-top": "20px"}
        ),
        
        html.Div(id="audio-div"),
        html.Div(id="comparison-div", style={"width": "95%", "margin": "auto"},
            children=[
                    # Images and Radio Question
                    html.H3("Transcriptions:"),
                    audio_trans_comp,
                    gen_comp,

                    # Question
                    html.H5("Select which transcription you prefer in each of the following dimensions.", \
                        style={"margin": '30px auto'}),
                    
                    html.Div(midi_comp_layout(), style={"display": "none", "width": "100%"}, id="midi-comp"),
                    html.Div(score_comp_layout(), style={"display": "none", "width": "100%"}, id="score-comp"),
                    dbc.Modal(
                            [
                                dbc.ModalBody("WARNING! Some questions have not been answered...", id="modal-warning-body"),
                                dbc.ModalFooter(
                                    html.Button("Close", n_clicks=0, id="close-warning-centered", style={
                                        "background-color": "#003366",   # dark blue
                                        "border-color": "#003366",
                                        "color": "white"
                                    })
                                ),
                            ],
                            id="warning-modal-centered",
                            centered=True,
                            is_open=False,
                    ),
            ] 
        ),
        nav_btns_component(),
    ]
)

# Useful for knowing if user reloads the page
# It will help in preventing the execution of certain callbacks
@callback(
    Output("reload", "data"),
    Input("reload", "data")
)
def reload(reload):
    # default value of reload is 0, if we reload the page, it
    # resets its value to 0; otherwise it is greater than 0
    return reload + 1

# Callback for going to next question
@callback(
    Output("max-btn-timestamp", "data"),
    Output("selection-changes", "data"),
    Output("last-click-next", "data"),
    Output("dimension-status", "data"),
    Output('current-question', 'data'),
    Output("next-btn", "children"),
    Output("finished", "data"), 
    Input("next-btn", "n_clicks_timestamp"),
    Input("back-btn", "n_clicks_timestamp"),
    State("current-question", "data"),
    State("last-click-next", "data"),
    State("midi-comp", "children"),
    State("score-comp", "children"),
    State("questions", "data"),
    State("max-btn-timestamp", "data"),
    State('reload', 'data'),
    prevent_initial_call=True
)
def go_to_next_question(n_clicks_front_ts, n_clicks_back_ts, current_idx,\
        last_click_next, midi_comp, score_comp, questions, max_ts, reload):
    
    if not reload and current_idx > -1:
        # if reload is 0 and current_idx is not -1, it means page has been reloaded!!
        raise PreventUpdate
    
    if n_clicks_front_ts is not None and n_clicks_front_ts > last_click_next:
        last_click_next = n_clicks_front_ts
    
    # Track what button was clicked more recently (Next btn or Back btn)
    # using max-btn-timestamp
    if n_clicks_front_ts is not None and n_clicks_front_ts > max_ts:
        max_ts = max(max_ts, n_clicks_front_ts)
    elif n_clicks_back_ts is not None and n_clicks_back_ts > max_ts:
        max_ts = max(max_ts, n_clicks_back_ts)
        current_idx = max(0, current_idx -1)

        # If back button is clicked, dim_status is True and we shouldn't be storing
        # any user input
        return max_ts, {}, last_click_next, True, current_idx, "Next", False
    
    if current_idx == -1:
        return max_ts, {}, last_click_next, True, 0, "Next", False
    
    # Get the current question before going to the next
    # We do this to see if the user has filled all requirements
    # for the current question...
    q = questions[current_idx]
    quest_type = q["type"]

    next_idx = current_idx
    if current_idx < 0:
        next_idx = 0
    elif current_idx >= len(questions) - 1:
        next_idx = len(questions) - 1
    else:
        next_idx += 1

    if next_idx == len(questions) - 1:
        next_btn_str = "Submit"
    else:
        next_btn_str = "Next"

    # Store user's selections and track if
    # he answers everything for each dimension of interest
    selections  = {}
    dim_status = False
    
    # Record current choices
    if quest_type == "MIDI":
        temp_dict = {}
        for i, child in enumerate(midi_comp):
            if i%2 != 0:
                # It means it is an horizontal line
                continue

            child = child["props"]["children"]
            label, radio_choice = child[0]["props"]["children"][0]["props"]["children"], child[1]["props"]["value"]
            if radio_choice is None:
                continue
            temp_dict[label] = radio_choice

        if temp_dict and len(temp_dict.keys()) == NUM_DIM_MIDI:
            selections[str(current_idx)] = temp_dict
            dim_status = True
    elif quest_type == "SCORE":
        temp_dict = {}
        for i, child in enumerate(score_comp):
            if i%2 != 0:
                # It means it is an horizontal line
                continue
            child = child["props"]["children"]
            label, radio_choice = child[0]["props"]["children"][0]["props"]["children"], child[1]["props"]["value"]
            if radio_choice is None:
                continue
            temp_dict[label] = radio_choice
        
        if temp_dict and len(temp_dict.keys()) == NUM_DIM_SCORE:
            selections[str(current_idx)] = temp_dict
            dim_status = True

    if current_idx == len(questions) - 1:
        if dim_status:
            return max_ts, selections, last_click_next, dim_status, current_idx, next_btn_str, True
        else:
            return max_ts, selections, last_click_next, dim_status, current_idx, next_btn_str, False
        
    if dim_status:
        return max_ts, selections, last_click_next, dim_status, next_idx, next_btn_str, False
    return max_ts, selections, last_click_next, dim_status, current_idx, next_btn_str, False

# Callback for rendering the radio items
@callback(
    Output("midi-comp", "style"),
    Output("score-comp", "style"),
    Output("q-type", "data"),
    Input("current-question", "data"),
    State("questions", "data"),
)
def render_radio(idx, questions):
    q = questions[idx] if idx < len(questions) else questions[-1]
    quest_type = q["type"]
    midi_comp_style = {"display": "none"}
    score_comp_style = {"display": "none"}
    
    if quest_type == "MIDI":
        midi_comp_style["display"] = "block"
    elif quest_type == "SCORE":
        # it means quest_type is score
        score_comp_style["display"] = "block"
    return midi_comp_style, score_comp_style, quest_type


# Callback for updating audio and comparison components
@callback(
    Output("audio-div", "children"),
    Output("img-1", "src"),
    Output("img-2", "src"),
    Output("midi-comp", "children"),
    Output('score-comp', 'children'),
    Output('prev-question', "data"),
    Output("aud-trans-1", "src"),
    Output("aud-trans-2", "src"),
    Input("current-question", "data"),
    State("user-selections", "data"),
    State("questions", "data"),
    State("order", "data"),
    State("prev-question", "data")
)
def update_question(idx, selections, questions, order, prev_q):
    # if the question's current index is equal to our previous index, do nothing
    if idx == prev_q:
        raise PreventUpdate
    
    prev_q = idx
    order = json.loads(order)
    q = questions[idx] if idx < len(questions) else questions[-1]
    audio_path = q["refAddress"]
    aTrans = q["aTrans"]
    bTrans = q["bTrans"]
    aTrans_audio = q["aTrans_audio"]
    bTrans_audio = q["bTrans_audio"]
    quest_type = q["type"]
    image_paths = [aTrans, bTrans]
    image_paths = sorted(image_paths, key=lambda x: order[idx][image_paths.index(x)])
    audio_trans_paths = [aTrans_audio, bTrans_audio]
    audio_trans_paths = sorted(audio_trans_paths, key=lambda x: order[idx][audio_trans_paths.index(x)])
    midi_comps = midi_comp_layout()
    score_comps = score_comp_layout()
    if str(idx) in selections:
        # if the user has already selected an option before,
        # simply load the state
        val = selections[str(idx)]
        if val and quest_type == "MIDI":
            # update the radio button states
            for i, item in enumerate(midi_comps):
                if i%2 != 0:
                    continue

                radioLabel = item.children[0].children[0].children
                #radioLabel = item.children[0].children[0].children[0].children
                #radioLabel = item.children[0].children
                radioItem = item.children[1]
                radioItem.value = val[radioLabel]
        elif val and quest_type == "SCORE":
            for i, item in enumerate(score_comps):
                if i%2 != 0:
                    continue

                radioLabel = item.children[0].children[0].children
                #radioLabel = item.children[0].children[0].children[0].children
                #radioLabel = item.children[0].children
                radioItem = item.children[1]
                radioItem.value = val[radioLabel]  

    audio = audio_component(audio_path)
    return audio, image_paths[0], image_paths[1], midi_comps, score_comps, prev_q, audio_trans_paths[0], audio_trans_paths[1]

# Callback for progress bar
@callback(
    Output("progress-bar", "value"),
    Input("current-question", "data"),
    State("questions", "data")
)
def update_progress(idx, questions):
    total = len(questions)
    progress = int((idx + 1) / total * 100)
    return progress

# Callback for Modal Warning
@callback(
    Output("warning-modal-centered", "is_open"),
    Output("modal-warning-body", "children"),
    Input("dimension-status", "data"),
    Input("close-warning-centered", "n_clicks"),
    State("next-btn", "children"),
    State("next-btn", "n_clicks_timestamp"),
    State("last-click-next", "data"),
    State("dimension-status", "data"),
    State("finished", "data"),
    State("warning-modal-centered", "is_open"),
    State("current-question", "data"),
    prevent_initial_call=True
)
def modal_warning(_, close_modal, next_btn_str, next_ts, last_click_next, dim_status, finished, is_open, current_idx):
    # We start the page with current_idx as -1 and so do nothing 
    # in this case
    if current_idx == -1:
        raise PreventUpdate
    
    if is_open:
        return False, ""
    
    if dim_status:
        # This means we are good to go
        return False, ""
    else:
        return True, "WARNING! Choose preferences for all dimensions before proceeding."

# # Store resposes
@callback(
    Input("finished", "data"),
    Input("user-selections", "data")
)
def save_responses(finished, selections):
    if not finished:
        PreventUpdate
    
    add_response(selections, user_id=current_user.get_id())
    # with open("responses.json", "w") as f:
    #     json.dump(selections, f, indent=4)
