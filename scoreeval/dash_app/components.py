from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

# Set number of dimensions for MIDI and Score
# We add 2 for likert based questions
NUM_DIM_MIDI = 5 + 2
NUM_DIM_SCORE = 8 + 2 
MIDI_LABELS = ['Draft', 'Faithfulness to Ref. Audio', 'Pitch', 'Rhythm', 'Harmony']
SCORE_LABELS = ['Draft', 'Faithfulness to Ref. Audio', 'Pitch', 'Metrical Alignment', 'Note Duration',\
                'Voice', 'Harmony', 'Notation']

TOOLTIP_TEXT_MIDI = {
    "Faithfulness to Ref. Audio": "Overall correspondence to the reference audio, including notes, structure, phrasing, and musical detail.",
    "Draft": "Preferred starting point for further editing and refinement.",
    "Pitch": "Accuracy of the pitch content relative to the reference audio.",
    "Rhythm": "How well the rhythmic feel and timing match the reference audio.",
    "Harmony": "Harmonic accuracy in terms of notes played simultaneously.",
}

TOOLTIP_TEXT_SCORE = {
    "Faithfulness to Ref. Audio": "Overall faithfulness to the reference audio.",
    "Draft": "Preferred starting point for refinement and editing.",
    "Pitch": "Accuracy of the pitch content relative to the reference audio.",
    "Harmony": "How well the key and chord structure reflect the harmonic essence of the piece?",
    "Voice": "How effectively are distinct monophonic voices (perceptual streams) separated and represented?",
    "Metrical Alignment": "How convincingly does the notated meter reflect the perceived metrical structure of the reference piece?",
    "Note Duration": "Transcription preference with regard to the relative duration of the predicted notes. Please ignore the effects of tempo.",
    "Notation": "Transcription preference based on these three notational aspects: staff assignment, pitch spelling, and stem directions."
}


TOOLTIP_IDS = {
    "Faithfulness to Ref. Audio": "target-overall",
    "Pitch": "target-pitch",
    "Rhythm": "target-rhythm",
    "Harmony": "target-harmony",
    "Voice": "target-voice",
    "Draft": "target-draft",
    "Metrical Alignment": "target-metrical-alignment",
    "Note Duration": "target-note-duration",
    "Notation": "target-notation"
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

    radio_items = dcc.RadioItems(
            id=radio_id,
            options=[{"label": f"{mapping[i]}", "value": str(i)} for i in range(5)],
            labelStyle={"display": "grid"},
            style={
            },
            className="likert-class",
            value=None
        )
    return html.Div(radio_items, className="likert-div")


def audio_component(audio_path: str) -> html.Div:
    component = html.Div(
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
        dbc.Col(html.Img(id='img-1', style={"border": "1px solid #ccc"}), xs=12, md=6, style={"padding": "0.5rem"}),
        dbc.Col(html.Img(id='img-2', style={"border": "1px solid #ccc"}), xs=12, md=6, style={"padding": "0.5rem"})
    ], style={"margin-bottom": "1.563rem"}, class_name="g-0") #g-0 classname removes negative margins + col padding that can cause overflow

audio_trans_comp = dbc.Row([
        dbc.Col(html.Audio(controls=True, id='aud-trans-1'), xs=12, md=6, style={"padding": "0.5rem"}),
        dbc.Col(html.Audio(controls=True, id='aud-trans-2'), xs=12, md=6, style={"padding": "0.5rem"}),
    ], style={"margin-bottom": "0.938rem"}, class_name="g-0")

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
                    style={"padding-right": "0.5rem"}
                ),
                dbc.Col(
                    html.Button("Next", id="next-btn", style={
                        "background-color": "#003366",   # dark blue
                        "border-color": "#003366",
                        "color": "white"
                    }),
                    width="auto",
                    style={"padding-left": "0.5rem"}
                ),
            ],
            justify="center", align="center",
            className="g-0 flex-wrap",
            ),
            style={ "margin": "2.5rem auto 1.25rem auto", "text-align": "center"},
            id="nav-btn-comp",
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
    
    if suffix=="midi":
        label_tooltip = html.Div(
            [
                html.Div(
                    label, style={"font-weight": "bold", "font-size": "large", "width": "100%"},
                    id=tp_id
                ),
                dbc.Tooltip(
                    TOOLTIP_TEXT_MIDI[label],
                    target=tp_id,
                    placement="top",
                ),
            ],
            style={"width": "5%"}
        )
    else:
        label_tooltip = html.Div(
            [
                html.Div(
                    label, style={"font-weight": "bold", "font-size": "large", "width": "100%"},
                    id=tp_id
                ),
                dbc.Tooltip(
                    TOOLTIP_TEXT_SCORE[label],
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
                                ))
            midi_comps.append(html.Hr(style={'border': '1px solid', "borderColor": "#000000", "opacity": "unset"}))
            midi_comps.append(item)
            #midi_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-bottom': '0px', \
            #                                 'margin-top': "20px", "borderColor": "#000000", "opacity": "unset"})),
            midi_comps.append(html.H5("How difficult was it to make a selection?"))
            midi_comps.append(html.Hr(style={}, className="likert-hr")),
            midi_comps.append(likert_component("likert-midi-draft"))
            continue
        elif idx == 1:
            midi_comps.append(html.H5("Select which transcription you prefer in each of the following dimensions.", \
                                style={"margin-top": "3.75rem"}))
            midi_comps.append(html.Hr(style={'border': '1px solid', "borderColor": "#000000", "opacity": "unset"}))
            midi_comps.append(item)
            midi_comps.append(html.Hr())
        elif idx == len(midi_list_comps) - 1:
            midi_comps.append(item)
            #midi_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-bottom': '0px', \
            #                                 'margin-top': "20px", "borderColor": "#000000", "opacity": "unset"})),
            midi_comps.append(html.H5("How difficult was it to make your selection?"))
            midi_comps.append(html.Hr(style={}, className="likert-hr")),
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
                                ))
            score_comps.append(html.Hr(style={'border': '1px solid', "borderColor": "#000000", "opacity": "unset"}))
            score_comps.append(item)
            #score_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-bottom': '0px', \
            #                                 'margin-top': "20px", "borderColor": "#000000", "opacity": "unset"})),
            score_comps.append(html.H5("How difficult was it to make a selection?"))
            score_comps.append(html.Hr(style={}, className="likert-hr")),
            score_comps.append(likert_component("likert-score-draft"))
            continue
        elif idx == 1:
            score_comps.append(html.H5("Select which transcription you prefer in each of the following dimensions.", \
                                style={ "margin-top": "3.75rem"}))
            score_comps.append(html.Hr(style={'border': '1px solid', "borderColor": "#000000", "opacity": "unset"}))
            score_comps.append(item)
            score_comps.append(html.Hr())
        elif idx == len(score_list_comps)-1:
            #score_comps.append(html.Hr(style={'border': '1px solid', 'width': '50%', 'margin-bottom': '0px', \
            #                                 'margin-top': "20px", "borderColor": "#000000", "opacity": "unset"})),
            score_comps.append(item)
            score_comps.append(html.H5("How difficult was it to make your selections?"))
            score_comps.append(html.Hr(style={}, className="likert-hr")),
            score_comps.append(likert_component("likert-score-trans"))
        else:
            score_comps.append(item)
            score_comps.append(html.Hr())
                
    return score_comps
