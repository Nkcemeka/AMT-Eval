"""
    Filename: questions.py
    Description: This file contains functionality for 
                 generating questions for the study and 
                 storing in a json file
"""
from pathlib import Path
import shutil
import json


CURRENT_DIR = Path(__file__).resolve().parent
MIDI_EXAMPLES_DIR = CURRENT_DIR / "./data/midi_examples/pred_midi"
STORE_DIR = CURRENT_DIR / "dash_app/assets"
MIDI_STORE_DIR = STORE_DIR / "midi_data"
SCORE_STORE_DIR = STORE_DIR / "score_data"
AUDIO_STORE_DIR = STORE_DIR / "audio"
PAUDIO_STORE_DIR = STORE_DIR / "paud_data"

# Create store directories if they don't exist
Path(MIDI_STORE_DIR).mkdir(exist_ok=True, parents=True)
Path(SCORE_STORE_DIR).mkdir(exist_ok=True, parents=True)
Path(AUDIO_STORE_DIR).mkdir(exist_ok=True, parents=True)
Path(PAUDIO_STORE_DIR).mkdir(exist_ok=True, parents=True)

questions_dict = {"questions": []}

def gen_questions(midi_json: str, score_json: str):
    with open(midi_json, "r") as me:
        midi_exs = json.load(me)
    
    with open(score_json, "r") as se:
        score_exs = json.load(se)

    # process the MIDI questions
    for i, mf in enumerate(midi_exs):
        assert Path(mf["audio"]).exists(), "Ground truth audio does not exist."
        assert Path(mf["images"][0]).exists() and Path(mf["images"][1]).exists(), "One/Both of the images do not exist."
        assert Path(mf["audio_trans"][0]).exists() and Path(mf["audio_trans"][1]).exists(),\
              "One/Both of the transcribed audios do not exist."
        assert mf["type"] == "MIDI", "Type should be MIDI"

        shutil.copy2(mf["audio"], AUDIO_STORE_DIR)
        shutil.copy2(mf["images"][0], MIDI_STORE_DIR)
        shutil.copy2(mf["images"][1], MIDI_STORE_DIR)
        shutil.copy2(mf["audio_trans"][0], PAUDIO_STORE_DIR)
        shutil.copy2(mf["audio_trans"][1], PAUDIO_STORE_DIR)

        q_dict = {"audio": f"assets/audio/{Path(mf["audio"]).stem+".wav"}",\
                  "images": [f"assets/midi_data/{Path(mf["images"][0]).stem+".png"}", \
                            f"assets/midi_data/{Path(mf["images"][1]).stem+".png"}"],
                  "audio_trans": [f"assets/paud_data/{Path(mf["audio_trans"][0]).stem+".wav"}",\
                        f"assets/paud_data/{Path(mf["audio_trans"][1]).stem+".wav"}"],
                  "type": "MIDI"}
        
        questions_dict["questions"].append(q_dict)

    
    for i, sf in enumerate(score_exs):
        print(score_exs[i])
        assert Path(sf["audio"]).exists(), "Ground truth audio does not exist."
        assert Path(sf["images"][0]).exists() and Path(sf["images"][1]).exists(), "One/Both of the images do not exist."
        assert Path(sf["audio_trans"][0]).exists() and Path(sf["audio_trans"][1]).exists(),\
              "One/Both of the transcribed audios do not exist."
        assert sf["type"] == "SCORE", "Type should be SCORE"

        shutil.copy2(sf["audio"], AUDIO_STORE_DIR)
        shutil.copy2(sf["images"][0], SCORE_STORE_DIR)
        shutil.copy2(sf["images"][1], SCORE_STORE_DIR)
        shutil.copy2(sf["audio_trans"][0], PAUDIO_STORE_DIR)
        shutil.copy2(sf["audio_trans"][1], PAUDIO_STORE_DIR)

        q_dict = {"audio": f"assets/audio/{Path(sf["audio"]).stem+".wav"}",\
                  "images": [f"assets/score_data/{Path(sf["images"][0]).stem+".png"}", \
                            f"assets/score_data/{Path(sf["images"][1]).stem+".png"}"],
                  "audio_trans": [f"assets/paud_data/{Path(sf["audio_trans"][0]).stem+".wav"}",\
                        f"assets/paud_data/{Path(sf["audio_trans"][1]).stem+".wav"}"],
                  "type": "SCORE"}
        
        questions_dict["questions"].append(q_dict)
    

# Generate questions and create JSON file
gen_questions("data/midi_examples/midi_samples.json", "data/score_examples/score_samples.json")
with open(str(STORE_DIR/"questions.json"), "w", encoding="utf-8") as f:
    json.dump(questions_dict, f, indent=4)
