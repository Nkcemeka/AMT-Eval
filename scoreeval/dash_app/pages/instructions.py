from dash import html
import dash
from dash import dcc 
from pathlib import Path

dash.register_page(__name__, path="/instructions")  # instructions page
# Get the path of this file
FILE_PATH = Path(__file__).resolve().parent
IMAGE_DIR = FILE_PATH / "../assets/ui_imgs"

md = f""" 
# Guidelines for User Study

Thank you for your decision to participate in this study. The goal of this research is to develop tools 
that **support musicians in the transcription process**, making transcription faster, less tedious, 
and more reliable. These tools are **not intended to replace musicians or musical judgment**, but to 
assist expert musicians by providing useful transcription drafts and alternatives.

Your expertise as a trained musician is essential. 
We are interested in how automatically generated transcriptions align with professional musical expectations, 
and how useful they are in real transcription workflows.

---
## The UI
The UI has been designed to make your experience simple and straightforward. We will quickly go through various
components you should be aware of.

**The Progress Bar**: This is a visual aid that gives you a rough idea of how close you are to finishing the study.  
![alt text](assets/ui_imgs/progress_bar.png)  
*Figure 1: Progress bar indicating study completion.*

**The Reference Audio Component**: This component allows you listen to the raw audio we wish to transcribe. To listen 
to the audio, simply press the play button.  
![alt text](assets/ui_imgs/ref_audio.png)  
*Figure 2: Reference Audio.*

**Transcription Pairs**: Each question comes with a pair of transcriptions which you have to evaluate. 
These transcriptions are presented as score or piano roll images.
Above each transcription image is an audio component. Each audio component is an audio rendition 
of the score or MIDI transcription under consideration.

![alt text](assets/ui_imgs/trans.png)  
*Figure 3a: Score Transcription Pairs*

![alt text](assets/ui_imgs/trans_midi.png)  
*Figure 3b: MIDI Transcription Pairs*

**Dimensions**: For each question, you will select the transcription you prefer under several dimensions. If you forget
what a dimension represents, simply hover over the name to get a helpful popup. For this section, you are not 
tasked with selecting a perfect transcription, but **which of the presented pair better satisfies the criterion 
referred to by each dimension**.

![alt text](assets/ui_imgs/dimensions.png)  
*Figure 4: Dimensions*

**Navigation Buttons**: At the end of the page, you have navigation buttons for each question. Note that you can only
move forward after selecting your **preferences for ALL dimensions**.  
![alt text](assets/ui_imgs/nav_btns.png)  
*Figure 5: Navigation Buttons*

---

### Your Role as an Expert Evaluator

In this study, you will act as an expert musical evaluator. There are no correct or incorrect answers. 
We are interested in your **musical judgment**, informed by your training, experience, and internal representation of the music.

You are encouraged to:
- Listen to the reference audio as many times as needed
- Analyze the music internally and mentally transcribe it
- Evaluate the symbolic transcriptions presented as images. You are provided to the audio renditions to aid you, but we advise you do not
rely primarily on it; instead base your preferences mainly on what you see.
- Select which transcription you prefer under the listed dimensions.

---

### Using the Reference Audio

For all questions, a **reference audio recording** is provided.

You are encouraged to:
- Listen to the audio multiple times
- Use the audio as your primary reference
- Base your judgments on how well each transcription represents the musical content of the reference audio

---

### Evaluating Suitability as a Starting Draft

All questions present a dimension that ask you to evaluate the **suitability of a transcription as a starting draft**.

In this context, imagine that you are beginning a manual transcription task. Consider:
- How much correction would be required
- Whether the transcription captures the essential musical structure
- Whether the presented score or piano roll would meaningfully reduce transcription effort

Please note that a transcription does **not** need to be correct or close to the reference to be a good starting draft.
Feel free to select what you will prefer to start with if you have to choose one as a basis for your work.

---

### General Guidelines

- Trust your musical instincts and professional standards
- Take your time: careful listening and evaluation are encouraged
- If unsure, choose the option that feels **more musically reasonable or useful**

---

### Thank You

Once again, thank you for the decision to participate. We strongly believe AI should be used to augment the workflow of musicians
rather than replace it. Your input directly contributes to the development of human-centered music transcription tools.
Thank you for taking the time to share your opinions.
"""

layout = html.Div(
    style={"text-align": "center", "padding": "50px"},
    children=[
        dcc.Markdown(
            md, style={"text-align": "justify"}
        ),
    ]
)
