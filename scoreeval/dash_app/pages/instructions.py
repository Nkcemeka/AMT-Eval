from dash import html
import dash
from dash import dcc 
from pathlib import Path
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/instructions")  # instructions page
# Get the path of this file
FILE_PATH = Path(__file__).resolve().parent
IMAGE_DIR = FILE_PATH / "../assets/ui_imgs"

md = f""" 
# Guidelines for User Study

Thank you for your decision to participate in this study. The goal of this research is to develop tools 
that **support musicians in the transcription process**; making transcription faster, less tedious, 
and more reliable. These tools are **not intended to replace musicians or musical judgment**, but to 
assist expert musicians by providing useful transcription drafts and alternatives.

Your expertise as a trained musician is essential. 
We are interested in how automatically generated transcriptions align with professional musical expectations, 
and how useful they are in real transcription workflows. Before proceeding, we **strongly recommend** 
using a *Chrome or Brave Browser* on a laptop or desktop computer.  This will ensure the best possible user experience 
as you navigate through the study.

---
## The UI
The User Interface (UI) has been designed to make your experience simple and straightforward. We will quickly go through various
components you should be aware of.

**The Progress Bar**: This is a visual aid that gives you a rough idea of how close you are to finishing the study.  
![alt text](assets/ui_imgs/progress_bar.png)  
*Figure 1: Progress bar indicating study completion.*

**The Reference Audio Component**: This component allows you listen to the raw audio we wish to transcribe. To listen 
to the audio, simply press the play button.  
![alt text](assets/ui_imgs/ref_audio.png)  
*Figure 2: Reference Audio.*

**Transcription Pairs**: We have a number of transcription pairs we want you to evaluate. 
These transcriptions are presented as score/piano roll images and are generated from two different music transcription 
models.
Above each transcription image is an audio component. The audio component is an acoustic rendition 
of the score or MIDI transcription under consideration. For the piano roll images, the notes are layered 
with the spectrogram information. This allows you view the notes and also the prominent frequencies
in the signal.

![alt text](assets/ui_imgs/trans.png)  
*Figure 3a: Score Transcription Pairs*

![alt text](assets/ui_imgs/trans_midi.png)  
*Figure 3b: MIDI Transcription Pairs*

**Dimensions**: For each presented transcription pair, you will select the transcription you prefer under several dimensions, 
which will be defined more extensively below. 
If you forget what a dimension represents during the study, simply hover over the name to get a helpful popup. For this section, you are 
tasked with selecting **which of the presented transcription pair better satisfies the criterion 
referred to by each dimension**.

![alt text](assets/ui_imgs/dimensions.png)  
*Figure 4: Dimensions*

**Selection Difficulty**: For each transcription pair, you will be asked to rate the selection difficulty for two different sections: 
appropriateness of a transcription as a starting draft and its quality under several musical dimensions.

![alt text](assets/ui_imgs/difficulty.png)  
*Figure 5: Selection Difficulty*

The difficulty level increases from left to right as shown in the image above following a Likert scale approach. 
For example, _Very Easy_ means you found the selection process to be an easy one; 
_Neutral_ means the selection preference was clear but required some level of thought and care; 
_Impossible_ means you found it hard to differentiate between the given transcription pairs. Your answer to this section
allow us to gauge the level of appropriateness for each transcription pair presented to you.

**Navigation Buttons**: At the end of the page, you have navigation buttons. Note that you can only
move forward after selecting your **preferences for ALL dimensions**.  Pressing the *Next* button saves the current 
state of your work; you can also logout and continue from where you stopped. It should be noted that pressing the
*Back* button, however, does not save any of your selections to the database.
![alt text](assets/ui_imgs/nav_btns.png)  
*Figure 6: Navigation Buttons*

---
### Dimension Definitions


##### MIDI Transcriptions
As a transcriber, you might be interested in the notes of a piece for many reasons. 
For instance, you might need to inspect the note content for specific passages where the music 
is ambiguous (e.g., fast runs, dense textures, clustered chords). 
Below are detailed definitions for each of the dimensions we want you to consider when evaluating MIDI transcriptions:

- **Draft**: If you had to choose one of the presented transcription pairs as a starting point for further editing, which would you prefer? 
This dimension reflects the practical workflow of a transcriber, where the output of an AI system serves as a foundation for refinement while 
still allowing room for improvement. Note that a transcription that is perfectly faithful to the reference piece is not necessarily 
the best draft. A strong draft may be imperfect, but structured in a way that supports efficient revision and further development.

- **Faithfulness to Ref. Audio**: Which transcription more accurately reflects the musical content of the reference audio? 
Consider the overall correspondence to what is heard, including notes, structure, phrasing, and musical detail.

- **Pitch**: This dimension focuses specifically on pitch accuracy. Which transcription more accurately represents the pitch content 
of the reference piece? Compare the transcriptions solely in terms of pitch, independent of other musical aspects.

- **Rhythm**: Which transcription better captures the rhythmic feel and timing of the reference audio?

- **Harmony**: Here, we want you to focus on notes played simultaneously and how this evolves over time. 
In other words, which transcription better represents the harmonic structure of the reference piece? 
A transcription may be harmonically complete (for example, including bass notes, inner voices, or upper-register notes) 
but introduce additional notes that are not present in the reference. Another may be sparser, 
omitting certain harmonic details while avoiding extraneous notes. In such cases, use your judgment to 
determine which transcription provides the more convincing harmonic representation. You may prioritize 
completeness or precision, whichever you prefer.

##### Score Transcriptions
As a transcriber, your ultimate goal is to produce a score that reflects the musical structure of the piece while 
incorporating your artistic judgment. The dimensions below capture different aspects of this process.

- **Draft**: Which transcription would you prefer to use as a starting point for your work? Again, note that a good transcription 
does not need to be a good draft; in some cases, a less faithful transcription may be easier to refine into a final version.

- **Faithfulness to Ref. Audio**: How well does the transcription capture the overall musical content—its structure, 
rhythm, phrasing, and the general flow of what is heard.

- **Pitch**: This dimension targets the accuracy of the pitch content in a transcription relative to the reference.

- **Metrical Alignment**: Here, the focus is on meter. Does the notated meter align with the metrical structure perceived from the audio? 
Even if multiple meters could describe the piece, which transcription offers the most convincing metrical representation?

- **Note Duration**: Which transcription more accurately represents **the relative durations of notes** as heard in the reference audio? 
For this dimension, ignore the effect of *tempo*. Instead, focus on the relative durations between notes and not the duration in absolute time.

- **Voice**: For this study, we define a voice as a monophonic perceptual stream of notes. By that, we mean note groupings in the score 
that seem to convey a musical phrase, thought, or idea. Which transcription better separates and represents distinct voices?

- **Harmony**: Harmony is strongly determined by the key of the piece and the chords that are used in the piece. Considering the key and the
 chords played in the score, which transcription do you think better captures the harmonic essence of the piece?

- **Notation**: Notation is a broad subject; it takes into consideration the assignment of notes to different staff, pitch spelling (C#/Db), 
beaming, ties, markings, and much more. For this study, we limit notation to staff assignment, stem direction, and pitch spelling. Based on 
these sub-aspects, what transcription do you prefer, notation-wise?


---

### Your Role as an Expert Evaluator

In this study, you will act as an expert musical evaluator. There are no correct or incorrect answers. 
We are interested in your **musical judgment**, informed by your training, experience, and internal representation of the music.

You are encouraged to:
- Listen to the reference audio as many times as needed
- Analyze the music internally and mentally transcribe it
- Evaluate the symbolic transcriptions presented as images. You are provided the audio renditions to aid you, but we advise you do not
rely primarily on it; instead, base your preferences mainly on what you see.
- Select which transcription you prefer under the listed dimensions.

---

### Using the Reference Audio

For all transcription pairs, a **reference audio recording** is provided.

You are encouraged to:
- Listen to the audio multiple times
- Use the audio as your primary reference
- Base your judgments on how well each transcription represents the musical content of the reference audio

---

### Evaluating Suitability as a Starting Draft

 Each comparison presents a dimension that asks you to evaluate the **suitability of a transcription as a starting draft**.

In this context, imagine that you are beginning a manual transcription task. Consider:
- How much correction would be required
- Whether the transcription captures the essential musical structure of the piece
- Whether the presented score or piano roll would meaningfully reduce transcription effort

Please note that a transcription does **not** need to be correct or close to the reference to be a good starting draft.
Feel free to select what you will prefer to start with if you have to choose one as a basis for your work.

---

### General Guidelines

- Trust your musical instincts and professional standards
- Take your time: careful listening and evaluation are encouraged
- If unsure, choose the option that feels **more musically reasonable or useful**

---
Once again, thank you for the decision to participate. We strongly believe artificially intelligent music tools should be 
used to augment the workflow of musicians rather than replace it. Your input directly contributes to the development of 
human-centered music transcription systems. Thank you for taking the time to share your opinions.
"""

layout = html.Div(
    style={"text-align": "center", "padding": "3.125rem"},
    children=[
        dcc.Markdown(
            md, style={"text-align": "justify", "width": "100%", "font-size": "large"}
        ),
        dcc.Link(dbc.Button("Proceed to Study", id="start-btn", style={
                "background-color": "#003366",   # dark blue
                "border-color": "#003366",
                "color": "white",
            }, size="lg"), href="/questions", style={"margin": "0.5rem"}),
    ]
)
