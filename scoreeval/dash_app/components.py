from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

# Set number of dimensions for MIDI and Score
NUM_DIM_MIDI = 5
NUM_DIM_SCORE = 6
MIDI_LABELS = ['Overall', 'Draft', 'Pitch', 'Rhythm', 'Harmony']
SCORE_LABELS = ['Overall', 'Draft', 'Pitch', 'Rhythm', 'Voice', 'Harmony']

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
    ], style={"margin-bottom": "15px"})

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
    for item in midi_list_comps:
        midi_comps.append(item)
        midi_comps.append(html.Hr())
    return midi_comps

# Get the Score components
def score_comp_layout():
    score_list_comps = list_comps(labels=SCORE_LABELS, suffix="score")
    score_comps = []
    for item in score_list_comps:
        score_comps.append(item)
        score_comps.append(html.Hr())
    return score_comps
