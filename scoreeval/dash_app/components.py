from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

# Set number of dimensions for MIDI and Score
# We add 2 for likert based questions
NUM_DIM_MIDI = 5 + 2
NUM_DIM_SCORE = 6 + 2 
MIDI_LABELS = ['Draft', 'Overall', 'Pitch', 'Rhythm', 'Harmony']
SCORE_LABELS = ['Draft', 'Overall', 'Pitch', 'Rhythm', 'Voice', 'Harmony']

TOOLTIP_TEXT = {
    "Overall": "The Overall dimension refers to transcription preference in terms of overall similarity to the reference",
    "Draft": "The Draft dimension indicates what you prefer as a starting transcription for your work.",
    "Pitch": "The Pitch dimension indicates what transcription is preferable in terms of pitch accuracy.",
    "Rhythm": "The Rhythm dimension indicates what transcription is preferable in terms of rhythm accuracy.",
    "Harmony": "The Harmony dimension indicates what transcription is preferable in terms of harmonic accuracy.",
    "Voice": "The Voice dimension indicates what transcription is preferable in terms of voice separation.",
}

TOOLTIP_IDS = {
    "Overall": "target-overall",
    "Pitch": "target-pitch",
    "Rhythm": "target-rhythm",
    "Harmony": "target-harmony",
    "Voice": "target-voice",
    "Draft": "target-draft"
}

def likert_component(radio_id):
    # Row of radio buttons
    mapping = {
        0: 'Very easy',
        1: 'Easy',
        2: 'Neutral',
        3: 'Difficult',
        4: 'Impossible'
    }

    return html.Div(dcc.RadioItems(
            id=radio_id,
            options=[{"label": f"{mapping[i]}", "value": str(i)} for i in range(5)],
            labelStyle={"display": "grid"},
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "width": "100%",
                "margin": "auto",
                "margin-left": "0px",
            },
            value=None
        ), style={"width": "50%"})


def audio_component(audio_path: str) -> html.Div:
    component = html.Div(
        style={"width": "95%", "margin": "auto", "padding": "10px"},
        children=[
            html.H3("Reference:"), 

            # Audio Player
            html.Audio(
                controls=True,
                src=f"{audio_path}",
            ),

            html.Hr()
        ]
    )
    return component

# general component for user selection
gen_comp = dbc.Row([
        dbc.Col(html.Img(id='img-1', style={"width": "100%", "border": "1px solid #ccc"})),
        dbc.Col(html.Img(id='img-2', style={"width": "100%", "border": "1px solid #ccc"}))
    ], style={"margin-bottom": "25px"})

audio_trans_comp = dbc.Row([
        dbc.Col(html.Audio(controls=True, id='aud-trans-1')),
        dbc.Col(html.Audio(controls=True, id='aud-trans-2')),
    ], style={"margin-bottom": "15px"})

def nav_btns_component() -> html.Div:
    nb_comp =  html.Div(
            dbc.Row([
                dbc.Col(
                    html.Button("Back", id="back-btn", style={
                        "background-color": "#003366",   # dark blue
                        "border-color": "#003366",
                        "color": "white"
                    }),
                    width="auto",
                ),
                dbc.Col(
                    html.Button("Next", id="next-btn", style={
                        "background-color": "#003366",   # dark blue
                        "border-color": "#003366",
                        "color": "white"
                    }),
                    width="auto"
                ),
            ],
            justify="center", align="center",
            ),
            style={ "margin-top": "40px", "margin-bottom": "20px", "text-align": "center"},
        )
    return nb_comp


def dimension_comp(label, radio_id, suffix) -> html.Div:
    radio_item = dcc.RadioItems(
                        id=radio_id,
                        options=[{"label": "", "value": "0"}, {"label": "", "value": "1"}], # 0 is left, 1 is right
                        labelStyle={"display": "inline-block"},  # horizontal layout
                        style={
                            "display": "flex",
                            "justify-content": "space-between",  # pushes first to left, last to right
                            "width": "50%",                      # adjust as needed
                            "margin": "auto",
                        },
                        value=None
    )

    # tooltip _id
    tp_id = TOOLTIP_IDS[label] + "-" + suffix

    # div_label = html.Div(label,
    #                 style={"font-weight": "bold", "width": "5%"})

    # div_label = html.Div(label,
    #                 style={"font-weight": "bold", "width": "5%"}, id=tp[0])
    
    label_tooltip = html.Div(
        [
            # html.P(
            #     children=[
            #         html.Span(label, style={"font-weight": "bold", "cursor": "pointer"}, id=tp_id)
            #     ]
            # ),
            html.Div(
                label, style={"font-weight": "bold", "width": "100%"},
                id=tp_id
            ),
            dbc.Tooltip(
                TOOLTIP_TEXT[label],
                target=tp_id,
                placement="top",
            ),
        ],
        style={"width": "5%"}
    )

    dim_comp = html.Div(
        style={"display": "flex", "justify-content": "space-between", "align-items": "center", \
               "width": "100%", "margin": "10px auto"},
        children=[label_tooltip, radio_item]
    )
    return dim_comp


def list_comps(labels=['Pitch', 'Rhythm', 'Harmony'], suffix="midi") -> list:
    """
        Generates a list of components
        for a MIDI or Score-based question
    """
    radio_ids = []
    for each in labels:
        radio_ids.append(each.lower()+'-'+'choice')
    res = []
    for each in zip(labels, radio_ids):
        label, radio_id = each[0], each[1]
        comp = dimension_comp(label, radio_id, suffix)
        res.append(comp)
    
    return res

def midi_comp_layout():
    # Get the MIDI components
    midi_list_comps = list_comps(labels=MIDI_LABELS)
    midi_comps = []
    for idx, item in enumerate(midi_list_comps):
        if idx == 0: # idx 0 is Draft
            midi_comps.append(html.H5("Select which transcription you prefer as a starting draft.", \
                                style={"margin-bottom": "30px"}))
            midi_comps.append(html.Hr(style={'border': '1px solid', "borderColor": "#000000", "opacity": "unset"}))
            midi_comps.append(item)
            #midi_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-bottom': '0px', \
            #                                 'margin-top': "20px", "borderColor": "#000000", "opacity": "unset"})),
            midi_comps.append(html.H5("How difficult was it to answer the question?"))
            midi_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-top': '0px', \
                                             "borderColor": "#000000", "opacity": "unset"})),
            midi_comps.append(likert_component("likert-midi-draft"))
            continue
        elif idx == 1:
            midi_comps.append(html.H5("Select which transcription you prefer in each of the following dimensions.", \
                                style={"margin-bottom": "30px", "margin-top": "60px"}))
            midi_comps.append(html.Hr(style={'border': '1px solid', "borderColor": "#000000", "opacity": "unset"}))
            midi_comps.append(item)
            midi_comps.append(html.Hr())
        elif idx == len(midi_list_comps) - 1:
            midi_comps.append(item)
            #midi_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-bottom': '0px', \
            #                                 'margin-top': "20px", "borderColor": "#000000", "opacity": "unset"})),
            midi_comps.append(html.H5("How difficult was it to answer the question?"))
            midi_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-top': '0px', \
                                            "borderColor": "#000000", "opacity": "unset"})),
            midi_comps.append(likert_component("likert-midi-trans"))
        else:
            midi_comps.append(item)
            midi_comps.append(html.Hr())
    return midi_comps

# Get the Score components
def score_comp_layout():
    score_list_comps = list_comps(labels=SCORE_LABELS, suffix="score")
    score_comps = []
    for idx, item in enumerate(score_list_comps):
        if idx == 0: # idx 0 is Draft
            score_comps.append(html.H5("Select which transcription you prefer as a starting draft.", \
                                style={"margin-bottom": "30px"}))
            score_comps.append(html.Hr(style={'border': '1px solid', "borderColor": "#000000", "opacity": "unset"}))
            score_comps.append(item)
            #score_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-bottom': '0px', \
            #                                 'margin-top': "20px", "borderColor": "#000000", "opacity": "unset"})),
            score_comps.append(html.H5("How difficult was it to answer the question?"))
            score_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-top': '0px', \
                                             "borderColor": "#000000", "opacity": "unset"})),
            score_comps.append(likert_component("likert-score-draft"))
            continue
        elif idx == 1:
            score_comps.append(html.H5("Select which transcription you prefer in each of the following dimensions.", \
                                style={"margin-bottom": "30px", "margin-top": "60px"}))
            score_comps.append(html.Hr(style={'border': '1px solid', "borderColor": "#000000", "opacity": "unset"}))
            score_comps.append(item)
            score_comps.append(html.Hr())
        elif idx == len(score_list_comps)-1:
            #score_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-bottom': '0px', \
            #                                 'margin-top': "20px", "borderColor": "#000000", "opacity": "unset"})),
            score_comps.append(item)
            score_comps.append(html.H5("How difficult was it to answer the question?"))
            score_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-top': '0px', \
                                            "borderColor": "#000000", "opacity": "unset"})),
            score_comps.append(likert_component("likert-score-trans"))
        else:
            score_comps.append(item)
            score_comps.append(html.Hr())
                
    return score_comps
