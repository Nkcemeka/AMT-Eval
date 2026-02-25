"""
    Filename: examples.py
    Description: This file contains functionality for extracting interesting 
                 examples for user study.
"""

# Import necessary modules
import argparse
import pretty_midi
from utils import merge_score, strip_tempo_markings
from pathlib import Path
import numpy as np
import librosa
import sys
sys.path.append("./extras")
import json
import torch
import snap2midi as s2m
from tqdm import tqdm
import pandas as pd
import soundfile as sf
from utils import trans, beyer_midi_xml, nakamura_inference, pm2s_inference, generate_piano_roll,\
      musescore_convert, musescore_convert_img, get_note_scores, compute_activation_metrics, \
      compute_mpteval, midi2audio, generate_piano_roll_spect, beyer_mscore_postprocess, \
      mv2h_eval, scoreMuster, scoreSim
from constants import KONG_CHECKPOINT, KONG_EXT_CONFIG, KONG_PEDAL_CHECKPOINT, HFT_CHECKPOINT, OAF_CHECKPOINT
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm, Normalize
import os
import itertools
import tempfile
import shutil
import matplotlib.patches as patches
from itertools import combinations
import random

class Band:
    """ 
        Data structure representing the 
        a collection of files that lie in
        the x% band in terms of absolute difference
        in F1 score!
    """
    def __init__(self, band):
        self.band = band
        self.trans = []
        self.f1_scores = []
    
    def add(self, trans1, trans2, trans1_f1, trans2_f1):
        trans1_img = trans1.replace("pred_midi", "pred_sproll").replace(".mid", ".png")
        trans2_img = trans2.replace("pred_midi", "pred_sproll").replace(".mid", ".png")
        trans1_aud = trans1.replace("pred_midi", "pred_audio").replace(".mid", ".wav")
        trans2_aud = trans2.replace("pred_midi", "pred_audio").replace(".mid", ".wav")
        self.trans.append((trans1_img, trans2_img, trans1_aud, trans2_aud))
        self.f1_scores.append((trans1_f1, trans2_f1))

    def __str__(self):
        return f"Band {int(self.band*100)}"
    
    def __repr__(self):
        return self.__str__()

class ExMidiBasic:
    def __init__(self, audioPath: str="./data/audio", midiPath: str = "./data/midi", \
                 midiScorePath: str="./data/midi_score", xmlScorePath: str="./data/xml_score",
                 ONSET_OFFSET_FLAG: bool=False, model1='kong', model2='transkun') -> None:
        """
            Instantiate ExMidiBasic. Accepted models are: [kong, transkun, oaf, hft]

            Args:
                audioPath (str): Path to directory containing audio files
                midiPath (str): Path to directory containing p-midi files
                midiScorePath (str): Path to directory containing midi score files
                xmlScorePath (str): Path to directory containing musicXML score files
                ONSET_OFFSET_FLAG (bool): If true, use F1 (Onset + Offset) instead of Onset only
                model1 (str): Name of first model to consider
                model2 (str): Name of second model to consider

            Returns:
                None
        """
        ACCEPTED_MODELS = set(['kong', 'transkun', 'oaf', 'hft'])
        if model1 not in ACCEPTED_MODELS or model2 not in ACCEPTED_MODELS:
            raise ValueError("Accepted models are: [kong, transkun, oaf, hft]")

        self.audio_files = sorted(Path(audioPath).glob("*.wav"))
        self.midi_files = sorted(Path(midiPath).glob("*.mid"))
        self.midi_score_files = sorted(Path(midiScorePath).glob("*.mid"))
        self.xml_score_files = sorted(Path(xmlScorePath).glob("*.musicxml"))

        # Assertion to ensure number of audio, midi, midi_score and xml_score files are the same
        assert len(self.audio_files) == len(self.midi_files) == len(self.midi_score_files) == len(self.xml_score_files), "\
            Number of audio, midi, midi_score and xml_score files are unequal!"
        
        self.inference = s2m.inference.Inference()
        self.ONSET_OFFSET_FLAG = ONSET_OFFSET_FLAG # if false, it uses onset-only F1 score
        self.STORE_PATH = "/home/nkcemeka/Documents/ismir2026/scoreeval/data/midi_examples"
        self.model1 = model1
        self.model2 = model2
    
    def get_midi_and_scores(self, audio_file, midi_file, model):
        if model == "transkun":
            midi_obj = trans(audio_file)
        elif model == "kong":
            midi_obj = self.inference.inference_kong(audio_file, checkpoint_note_path=KONG_CHECKPOINT, \
                        checkpoint_pedal_path=KONG_PEDAL_CHECKPOINT,\
                        filename=None, user_ext_config=KONG_EXT_CONFIG)
        elif model == "oaf":
            midi_obj = self.inference.inference_oaf(audio_file, checkpoint_path=OAF_CHECKPOINT, filename=None)
        elif model == "hft":
            midi_obj = self.inference.inference_hft(audio_file, checkpoint_path=HFT_CHECKPOINT, filename=None) 
        else:
            raise ValueError(f"Invalid model input: {model}")

        # Remove sustain from Transkun or any other output with sustain info
        for instrument in midi_obj.instruments:
            instrument.control_changes = [
                cc for cc in instrument.control_changes if cc.number != 64
            ]

        # We use the try-except clause for edge cases where measure is short or fast (1s)
        # and some model fails to predict anything for whatever weird reason
        score_dict = get_note_scores(midi_obj, str(midi_file))
        if self.ONSET_OFFSET_FLAG:
            score = score_dict["f_off"]
        else:
            score = score_dict["f"]
        torch.cuda.empty_cache()
        return midi_obj, score
    
    def gen_examples(self):
        scores_list_model1 = []
        scores_list_model2 = []
        interesting_files = []

        # Create self.STORE_PATH and necessary subdirectories if they do not exist
        Path(self.STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_audio").mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_midi").mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_proll").mkdir(parents=True, exist_ok=True) 
        Path(self.STORE_PATH + "/pred_sproll").mkdir(parents=True, exist_ok=True) # spectrogram piano roll

        for idx, (audio_file, midi_file, midi_score_file, xml_score_file) in tqdm(enumerate(zip(self.audio_files, \
            self.midi_files, self.midi_score_files, self.xml_score_files)), total=len(self.audio_files),\
            desc="Processing files"):

            # dictionary to store our interesting examples
            int_dict = {
                        'audio': str(audio_file),
                        'midi': str(midi_file),
                        'midi_score': str(midi_score_file), 
                        'xml_score': str(xml_score_file),
            }
            
            assert audio_file.stem == midi_file.stem, f"Audio and MIDI file names do not match: {audio_file.stem} != {midi_file.stem}"
            model1_filename = audio_file.stem + f"_{self.model1}"
            model2_filename = audio_file.stem + f"_{self.model2}"
            try:
                model1_midi, score1 = self.get_midi_and_scores(audio_file, midi_file, self.model1)
                model2_midi, score2 = self.get_midi_and_scores(audio_file, midi_file, self.model2)
            except:
                continue

            # Store scores in an arr
            int_dict[f'{model1_filename}_pmidi_f1'] = score1
            int_dict[f'{model2_filename}_pmidi_f1'] = score2
            scores_list_model1.append(score1)
            scores_list_model2.append(score2)
            
            # save MIDI files
            model1_midi.write(self.STORE_PATH + "/pred_midi/" + model1_filename + ".mid")
            model2_midi.write(self.STORE_PATH + "/pred_midi/" + model2_filename + ".mid")
            int_dict[f'{model1_filename}_pmidi'] = self.STORE_PATH + "/pred_midi/" + model1_filename + ".mid"
            int_dict[f'{model2_filename}_pmidi'] = self.STORE_PATH + "/pred_midi/" + model2_filename + ".mid"

            # Get audio for both MIDI
            SOUNDFONT_PATH = "/home/nkcemeka/Documents/snap/snap2midi/notebooks/soundfonts/MuseScore_General.sf2"
            audio_model1 = midi2audio(model1_midi, sf2_path=SOUNDFONT_PATH)
            audio_model2 = midi2audio(model2_midi, sf2_path=SOUNDFONT_PATH)
            sf.write(self.STORE_PATH + "/pred_audio/" + model1_filename + ".wav", audio_model1, 16000)
            sf.write(self.STORE_PATH + "/pred_audio/" + model2_filename + ".wav", audio_model2, 16000)

            # Get piano roll & spect. piano roll for both MIDI
            # for both MIDIs, get the min_pitch and max_pitch
            min_pitch = 108
            max_pitch = 21
            for i in range(2):
                if i == 0:
                    midi_obj = model1_midi
                else:
                    midi_obj = model2_midi
                for n in midi_obj.instruments[0].notes:
                    if n.pitch > max_pitch:
                        max_pitch = n.pitch
                    
                    if n.pitch < min_pitch:
                        min_pitch = n.pitch

            int_dict[f'min_pitch'] = min_pitch
            int_dict[f'max_pitch'] = max_pitch
            y, sr = librosa.load(str(audio_file), sr=16000)
            D = librosa.stft(y)
            spect = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr)[1:]

            for i in range(2):
                if i == 0:
                    midi_name = model1_filename
                    midi_obj = model1_midi
                else:
                    midi_name = model2_filename
                    midi_obj = model2_midi
                STORE_PATH = Path(self.STORE_PATH) / "pred_sproll" / f"{midi_name}.png"
                generate_piano_roll_spect(midi_obj, spect, STORE_PATH, freqs, sr, min_pitch=min_pitch, max_pitch=max_pitch)
                generate_piano_roll(midi_obj, str(Path(self.STORE_PATH) / f"pred_proll/{midi_name}.png"),\
                                    min_pitch=min_pitch, max_pitch=max_pitch)

            interesting_files.append(int_dict)
        
        if interesting_files:
            json_out = {
                'files': interesting_files
            }

            with open(f"{str(self.STORE_PATH)}/examples.json", "w", encoding="utf-8") as f:
                json.dump(json_out, f, indent=4)
                        

        fig, ax = plt.subplots()
        ax.scatter(scores_list_model1, scores_list_model2, color='red')
        ax.axhline(y=0.5)
        ax.axvline(x=0.5)
        # Split the top-right quadrant into 4 equal parts
        # ax.plot([0.75, 0.75], [0.5, 1.0], color='green', linewidth=2)
        # ax.plot([0.5, 1.0], [0.75, 0.75], color='green', linewidth=2)

        # draw lines for different absolute differences in F1 scores
        # realistically speaking (for the top upper quadrant, a diff_F > 0.5 is meaningless)
        # This won't show up in our plot because F1 score can't be greater than 1
        # since we are in the upper quadrant, the diff_f <=0.5
        diff_F = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        x = np.linspace(0.5, 1.0, 100)

        # generate our colormap
        norm = colors.Normalize(vmin=0, vmax=diff_F.max())
        cmap = cm.viridis

        for i in range(len(diff_F)-1):
            c_low = diff_F[i]
            c_high = diff_F[i+1]
            color = cmap(norm((c_high)))

            # upper band: x + c_low to x + c_high (we condsider region above line of y=x here)
            # for each x[i], we have a vertical line connecting
            # y1 and y2...
            y1 = x + c_low
            y2 = x + c_high

            # since we consider the upper left-right angled triangle here
            # we need to ensure y1 never exceeds 1 and y2 never goes below 0.5
            mask = (y1 <= 1.0) & (y2 >= 0.5)
            ax.fill_between(
                x[mask],
                np.clip(y1[mask], 0.5, 1.0),
                np.clip(y2[mask], 0.5, 1.0),
                color=color,
                alpha=0.4
            )

            # Lower band: x - c_high to x - c_low
            y1 = x - c_high
            y2 = x - c_low

            # since we consider the bottom-right right angled triangle here
            # we need to ensure y1 never exceeds 1 and y2 never goes below 0.5
            mask = (y2 >= 0.5) & (y1 <= 1.0)
            ax.fill_between(
                x[mask],
                np.clip(y1[mask], 0.5, 1.0),
                np.clip(y2[mask], 0.5, 1.0),
                color=color,
                alpha=0.4
            )

        # for c in diff_F:
        #     color = cmap(norm(c))
        #     y = x + c
        #     y_neg = x-c
        #     mask = (y >= 0.5) & (y <= 1.0)
        #     mask_neg = (y_neg >= 0.5) & (y_neg <= 1.0)
            # ax.plot(x[mask], y[mask], color=color, linestyle='--', linewidth=1, alpha=0.7)
            # ax.plot(x[mask_neg], y_neg[mask_neg], color=color, linestyle='--', linewidth=1, alpha=0.7)

        # Create a colorbar for the lines
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("|F₂ - F₁| band")

        ax.set_ylabel(f"F_{self.model2}")
        ax.set_xlabel(f"F_{self.model1}")
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(0, 1)
        ax.set_title(f"Scatter plot of F_{self.model2} against F_{self.model1}")
        fig.savefig(str(Path(self.STORE_PATH)/f"fig_{self.model1}_{self.model2}.png"))
    
    def compute_f12(self, midi1: pretty_midi.PrettyMIDI, midi2: pretty_midi.PrettyMIDI) -> float:
        """ 
            Compute the F1 scores between two models

            Args:
                midi1 (pretty_midi.PrettyMIDI): Midi1 pretty_midi object
                midi2 (pretty_midi.PrettyMIDI): Midi2 pretty_midi object
        """
        if self.ONSET_OFFSET_FLAG:
            # we can use any score fn. Using several is unnecesary
            # I only did that because I wanted to use different pedal
            # extension techniques
            f12 = get_note_scores(midi1, midi2)['f_off']
            f12_check = get_note_scores(midi2, midi1)['f_off']
        else:
            f12 = get_note_scores(midi1, midi2)['f']
            f12_check = get_note_scores(midi2, midi1)['f']
        
        assert f12 == f12_check, "The F1 scores should be symmetric between the two models"
        return f12

    def sample(self, json_file: str, NUM_EXAMPLES=10):
        """  
            Samples potential examples from each band.

            Args:
                json_file (str): Path to examples json file
                NUM_EXAMPLES (int): Number of examples to extract when sampling

            Algorithm:
                1. Get the number of questions in each band (10 to 50%). So for each band,
                we have a data structure that houses the path to the files for each band.
                2. Sort the bands by the number of questions
                3. num_ques_band = NUM_EXAMPLES/5
                4. starting from the band B with least number of questions, we ask: is |B| < num_quest_band?
                if yes, select all of |B| and recalculate num_questions_band for the rest; if no, randomly
                select num_quest_band and continue
                5. Store the final sampling selection in a file or sumn. 
            
            Returns:
                None
        """
        b10 = Band(0.1); b20 = Band(0.2); b30 = Band(0.3); b40 = Band(0.4); b50 = Band(0.5)
        
        # load json_file of examples
        with open(json_file, "r") as f:
            exs = json.load(f)["files"]

        for i, d in enumerate(exs):
            filename = Path(d["audio"]).stem
            model1_f1_tag = filename + f"_{self.model1}_pmidi_f1"
            model2_f1_tag = filename + f"_{self.model2}_pmidi_f1"
            abs_diff_f1 = abs(d[model1_f1_tag] - d[model2_f1_tag])
            trans1 = d[filename + f"_{self.model1}_pmidi"]
            trans2 = d[filename + f"_{self.model2}_pmidi"]

            if d[model1_f1_tag] < 0.5 or d[model2_f1_tag] < 0.5:
                continue

            if abs_diff_f1 <= b10.band:
                b10.add(trans1, trans2, d[model1_f1_tag], d[model2_f1_tag])
            elif abs_diff_f1 <= b20.band:
                b20.add(trans1, trans2, d[model1_f1_tag], d[model2_f1_tag])
            elif abs_diff_f1 <= b30.band:
                b30.add(trans1, trans2, d[model1_f1_tag], d[model2_f1_tag])
            elif abs_diff_f1 <= b40.band:
                b40.add(trans1, trans2, d[model1_f1_tag], d[model2_f1_tag])
            elif abs_diff_f1 <= b50.band:
                b50.add(trans1, trans2, d[model1_f1_tag], d[model2_f1_tag])
            else:
                raise RuntimeError("F1 score diff. should not exceed 50%!")
        
        current_num_bands = 5
        num_quest_band = NUM_EXAMPLES//current_num_bands
        sampled_questions = []
        band_list = [b10, b20, b30, b40, b50]

        # sort the band list based on the number of items it has (in increasing order)
        band_list.sort(key=lambda x: len(x.trans))

        for i in range(len(band_list)):
            band = band_list[i]
            len_band = len(band.trans)
            if len_band < num_quest_band:
                sampled_questions.extend(band.trans)
                NUM_EXAMPLES -= len_band
            else:
                samples = random.sample(band.trans, num_quest_band)
                sampled_questions.extend(samples)
                NUM_EXAMPLES -= num_quest_band
            
            if i != len(band_list)-1:
                current_num_bands -= 1
                num_quest_band = (NUM_EXAMPLES)//current_num_bands
                
        # Generate a hash for audio
        gt_hash = {}
        for aud in self.audio_files:
            stem = aud.stem
            gt_hash[stem] = str(aud)
        
        # save the list of sampled questions to a dict
        final_selection = []
        for each in sampled_questions:
            each_stem = "_".join(Path(each[0]).stem.split('_')[:-1])
            sample_dict = {}
            sample_dict["audio"] = gt_hash[each_stem]
            sample_dict["images"]= [each[0], each[1]]
            sample_dict["audio_trans"] = [each[2], each[3]]
            sample_dict["type"] = "MIDI"
            final_selection.append(sample_dict)

        with open(self.STORE_PATH + "/midi_samples.json", "w") as f:
            json.dump(final_selection, f)

class ExScore:
    """ 
        Generate examples for the MIDI
        to Score evaluation!
    """
    def __init__(self, audioPath: str="/home/nkcemeka/Documents/ismir2026/scoreeval/data/audio", \
                 midiPath: str = "/home/nkcemeka/Documents/ismir2026/scoreeval/data/midi", \
                 midiScorePath: str="/home/nkcemeka/Documents/ismir2026/scoreeval/data/midi_score", \
                 xmlScorePath: str="/home/nkcemeka/Documents/ismir2026/scoreeval/data/xml_score"):
        
        self.audio_files = sorted(Path(audioPath).glob("*.wav")) # I had 400 before
        self.midi_files = sorted(Path(midiPath).glob("*.mid"))
        self.midi_score_files = sorted(Path(midiScorePath).glob("*.mid"))
        self.xml_score_files = sorted(Path(xmlScorePath).glob("*.musicxml"))
        self.STORE_PATH = "/home/nkcemeka/Documents/ismir2026/scoreeval/data/score_examples"
        self.audioPath = audioPath
        self.midiPath = midiPath
        self.midiScorePath = midiScorePath
        self.xmlScorePath = xmlScorePath
        
    
    def gen_examples(self):
        # Create self.STORE_PATH and necessary subdirectories if they do not exist
        Path(self.STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_xml").mkdir(parents=True, exist_ok=True) 
        Path(self.STORE_PATH + "/pred_mscore").mkdir(parents=True, exist_ok=True) 
        Path(self.STORE_PATH + "/pred_img").mkdir(parents=True, exist_ok=True)
        example_files = []

        for idx, (audio_file, midi_file, midi_score_file, xml_score_file) in tqdm(enumerate(zip(self.audio_files, \
            self.midi_files, self.midi_score_files, self.xml_score_files)), total=len(self.audio_files),\
            desc="Processing files"):

            assert audio_file.stem == midi_file.stem, f"Audio and MIDI file names do not match: {audio_file.stem} != {midi_file.stem}"
            filename = str(midi_file.stem)
            beyer_mscore_path = str(Path(self.STORE_PATH) / f"pred_mscore/{filename}_beyer.mid")
            nak_mscore_path = str(Path(self.STORE_PATH) / f"pred_mscore/{filename}_nak.mid")
            pm2s_mscore_path = str(Path(self.STORE_PATH) / f"pred_mscore/{filename}_pm2s.mid")
            musescore_mscore_path = str(Path(self.STORE_PATH) / f"pred_mscore/{filename}_musescore.mid")
            beyer_xml_path = beyer_mscore_path.replace("pred_mscore", "pred_xml").replace("mid", "musicxml")
            nak_xml_path = nak_mscore_path.replace("pred_mscore", "pred_xml").replace("mid", "musicxml")
            pm2s_xml_path = pm2s_mscore_path.replace("pred_mscore", "pred_xml").replace("mid", "musicxml")
            musescore_xml_path = musescore_mscore_path.replace("pred_mscore", "pred_xml").replace("mid", "musicxml")

            ex_dict = {
                    'audio': str(audio_file),
                    'midi': str(midi_file),
                    'midi_score': str(midi_score_file), 
                    'xml_score': str(xml_score_file),
            }

            # Get the Quantized predictions and XML
            try:
                # Get quantized scores for pm2s and nakamura
                pm2s_inference(str(midi_file), pm2s_mscore_path)
                nakamura_inference(str(midi_file), nak_mscore_path)

                # Convert the mscores for nakamaura and pm2s to xml with musescore
                musescore_convert(pm2s_mscore_path, pm2s_xml_path)
                # For PM2S, strip the final score of tempo markings for rendering purposes
                strip_tempo_markings(pm2s_xml_path, pm2s_xml_path)

                # merge nakamura before conversion to xml
                musescore_convert(nak_mscore_path, nak_xml_path)
                merge_score(nak_xml_path, nak_xml_path)
                musescore_convert(nak_xml_path, nak_mscore_path)
                # For nakamura, strip the final score off the tempo information for rendering
                # purposes
                strip_tempo_markings(nak_xml_path, nak_xml_path)

                # For musescore, get the xml first
                musescore_convert(str(midi_file), musescore_xml_path)
                # now for musescore, get the mscore
                musescore_convert(musescore_xml_path, musescore_mscore_path)
                # now strip the xml of the tempo information for rendering processes
                strip_tempo_markings(musescore_xml_path, musescore_xml_path)

                # For beyer, get the XML first
                beyer_midi_xml(str(midi_file), beyer_xml_path)
                # Now get the mscore for beyer with musescore
                musescore_convert(beyer_xml_path, beyer_mscore_path)
                # For beyer, we will read the mscore and postprocess it
                beyer_mscore_postprocess(beyer_mscore_path, beyer_mscore_path)

                assert Path(beyer_mscore_path).exists()
                assert Path(beyer_xml_path).exists()
                assert Path(pm2s_mscore_path).exists()
                assert Path(pm2s_xml_path).exists()
                assert Path(nak_mscore_path).exists()
                assert Path(nak_xml_path).exists()
                assert Path(musescore_mscore_path).exists()
                assert Path(musescore_xml_path).exists()
            except:
                # if anything fails, remove the clutter
                if os.path.exists(beyer_xml_path):
                    os.remove(beyer_xml_path)
                
                if os.path.exists(beyer_mscore_path):
                    os.remove(beyer_mscore_path)

                if os.path.exists(nak_xml_path):
                    os.remove(nak_xml_path)
                
                if os.path.exists(nak_mscore_path):
                    os.remove(nak_mscore_path)
                
                if os.path.exists(pm2s_xml_path):
                    os.remove(pm2s_xml_path)
                
                if os.path.exists(pm2s_mscore_path):
                    os.remove(pm2s_mscore_path)
                
                if os.path.exists(musescore_mscore_path):
                    os.remove(musescore_mscore_path)
                
                if os.path.exists(musescore_xml_path):
                    os.remove(musescore_xml_path)
                continue
                
            # convert xmls to images for all models
            musescore_convert_img(beyer_xml_path, beyer_xml_path.replace("pred_xml", "pred_img").replace("musicxml", "png"))
            musescore_convert_img(nak_xml_path, nak_xml_path.replace("pred_xml", "pred_img").replace("musicxml", "png"))
            musescore_convert_img(musescore_xml_path, musescore_xml_path.replace("pred_xml", "pred_img").replace("musicxml", "png"))
            musescore_convert_img(pm2s_xml_path, pm2s_xml_path.replace("pred_xml", "pred_img").replace("musicxml", "png"))
            
            ex_dict['beyer'] = [beyer_mscore_path, beyer_xml_path]
            ex_dict['nakamura'] = [nak_mscore_path, nak_xml_path]
            ex_dict['musescore'] = [musescore_mscore_path, musescore_xml_path]
            ex_dict['pm2s'] = [pm2s_mscore_path, pm2s_xml_path]
            example_files.append(ex_dict)
        
        # save examples
        json_out = {
            'files': example_files
        }

        with open(f"{str(self.STORE_PATH)}/examples.json", "w", encoding="utf-8") as f:
            json.dump(json_out, f, indent=4)
        
        # Generate audio for all of the engravings
        self.gen_audio(f"{str(self.STORE_PATH)}/examples.json")
        model_list = ['beyer', 'pm2s', 'nakamura', 'musescore']
        model_pairs = list(combinations(model_list, 2))
        for pair in model_pairs:
            self.gen_plots(pair[0], pair[1])
    
    def gen_audio(self, json_file: str):
        """
            Generate audio files for every transcription!

            Args:
            -----
                json_file (str): Path to json file containing examples
        """
        STORE_DIR = Path(self.STORE_PATH) /"pred_audio"
        SOUNDFONT_PATH = "/home/nkcemeka/Documents/snap/snap2midi/notebooks/soundfonts/MuseScore_General.sf2"
        Path.mkdir(STORE_DIR, exist_ok=True, parents=True) # Create path to store audio files
        with open(json_file, "r") as f:
            exs = json.load(f)

        for each in tqdm(exs["files"]):
            for k in ['beyer', 'pm2s', 'musescore', 'nakamura']:
                midi_path = Path(each[k][0])
                midi_name = midi_path.stem
                aud_name = midi_name + ".wav"
                aud_path = STORE_DIR / aud_name
                midi_obj = pretty_midi.PrettyMIDI(str(midi_path))
                audio_arr = midi2audio(midi_obj, sf2_path=SOUNDFONT_PATH)
                # save the audio file
                sf.write(str(aud_path),audio_arr,16000)

    
    def gen_plots(self, model1: str, model2: str):
        """ 
            Generates plots based on the MV2H metric.
            Accepted model names are: ['beyer', 'musescore', \
                'nakamura', 'pm2s']

            Args:
                model1 (str): model1 name
                model2 (str): model2 name
        """
        MSCORE_PATH = Path(self.STORE_PATH) / "pred_mscore"
        #XML_PATH = Path(self.STORE_PATH) / "pred_xml"
        GT_DIR = Path("/home/nkcemeka/Documents/ismir2026/scoreeval/data/midi_score")
        # GT_DIR = Path("/home/nkcemeka/Documents/ismir2026/scoreeval/data/xml_score")
        GT_HASHMAP = {}
        for each in sorted(GT_DIR.glob("*.mid")):
            stem = each.stem
            GT_HASHMAP[stem] = str(each)

        #gt_files = Path(GT_DIR).glob("*.mid")
        model1_files = sorted(Path(MSCORE_PATH).glob(f"*{model1}*.mid"))
        model2_files = sorted(Path(MSCORE_PATH).glob(f"*{model2}*.mid"))
        # model1_files = sorted(Path(XML_PATH).glob(f"*{model1}*.musicxml"))
        # model2_files = sorted(Path(XML_PATH).glob(f"*{model2}*.musicxml"))
        #mus_score1 = []
        #mus_score2 = []
        mv2h_score1 = []
        mv2h_score2 = []

        assert len(model1_files) == len(model2_files), f"Files for {model1} and {model2} are not equal!"

        for path1, path2 in tqdm(zip(model1_files, model2_files), total=len(model1_files)):
            stem = "_".join(path1.stem.split("_")[:-1])
            gt_path = GT_HASHMAP[stem]
            
            try:
                assert Path(gt_path).exists(), f"{str(gt_path)} does not exist"
                assert Path(path1).exists(), f"{str(path1)} does not exist."
                assert Path(path2).exists(), f"{str(path2)} does not exist."
                score1 = mv2h_eval(gt_path, str(path1))['MV2H']
                score2 = mv2h_eval(gt_path, str(path2))['MV2H']
                print(score1, score2, score2 > score1)
            except:
                continue

            #score1_dict = scoreMuster(gt_path, str(path1))
            #score2_dict = scoreMuster(gt_path, str(path2))

            # if score1_dict is None or score2_dict is None:
            #     continue

            # score1 = 0
            # score2 = 0
            # for k in score1_dict.keys():
            #     score1 += score1_dict[k]
            #     score2 += score2_dict[k]
            
            # score1 /= len(score1_dict.keys())
            # score2 /= len(score2_dict.keys())
            # mus_score1.append(score1/100)
            # mus_score2.append(score2/100)
            mv2h_score1.append(score1)
            mv2h_score2.append(score2)

        fig, ax = plt.subplots()
        ax.scatter(mv2h_score1, mv2h_score2, color='red')
        ax.axhline(y=0.5)
        ax.axvline(x=0.5)
        ax.set_ylabel(f"MV2H_{model2}")
        ax.set_xlabel(f"MV2H_{model1}")
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(0, 1)
        ax.set_title(f"Scatter plot of MV2H_{model2} against MV2H_{model1}")
        fig.savefig(str(Path(self.STORE_PATH)/f"fig_{model1}_{model2}.png"))
        
        # fig, ax = plt.subplots()
        # ax.scatter(mus_score1, mus_score2, color='red')
        # ax.axhline(y=0.5)
        # ax.axvline(x=0.5)
        # ax.set_ylabel(f"Muster_{model2}")
        # ax.set_xlabel(f"Muster_{model1}")
        # ax.set_ylim(bottom=0, top=1)
        # ax.set_xlim(0, 1)
        # ax.set_title(f"Scatter plot of Muster_{model2} against Muster_{model1}")
        # fig.savefig(str(Path(self.STORE_PATH)/f"fig_{model1}_{model2}.png"))
    
    def sample(self, xlsx_file: str, NUM_EXAMPLES: int = 10):
        """ 
            Sample based on the segments that work for all
            calculated metrics.
        """
        sheet_names = ["Beyer", "Musescore", "PM2S", "Nakamura"]
        xlsx_objs = []
        piece_objs = []
        hash_objs = [{}, {}, {}, {}]
        final_hash = {}

        for sheet in sheet_names:
            xlsx_objs.append(pd.read_excel(xlsx_file, sheet_name=sheet, header=[0, 1]))
            piece_objs.append(xlsx_objs[-1][('piece', 'name')].dropna()[1:-1].reset_index(drop=True))
        
        for i, piece in enumerate(piece_objs):
            for each in piece:
                if each == "Avg":
                    continue
                items = Path(each).stem.split("_")
                key = items[0] + "_" + items[1]
                hash_objs[i][key] = each
        
        # keep pieces in all hash_objs
        for i, piece in enumerate(piece_objs):
            for each in piece:
                if each == "Avg":
                    continue
                items = Path(each).stem.split("_")
                key = items[0] + "_" + items[1]
                valid = (key in hash_objs[0]) & (key in hash_objs[1]) & (key in hash_objs[2]) & (key in hash_objs[3])
                if valid:
                    final_hash[key] = None
        
        # Get all the paths to the audio files in GTAUD path
        gt_hash = {}
        for aud, mid, mscore, xml_score in zip(self.audio_files, self.midi_files, self.midi_score_files, self.xml_score_files):
            assert aud.stem == mid.stem == mscore.stem == xml_score.stem, \
                "Audio, MIDI, Mscore and XML Score should be of the same audio"
            stem = aud.stem
            gt_hash[stem] = [str(aud), str(mid), str(mscore), str(xml_score)]

        # perform the sampling process
        sel_samples = random.sample(list(final_hash.keys()), NUM_EXAMPLES)
        final_selection = []
        IMG_PATH = Path(self.STORE_PATH) / "pred_img"
        for s in sel_samples:
            i, j = random.sample(range(4), 2)
            trans1_img = str(IMG_PATH / hash_objs[i][s]) + ".png"
            trans2_img = str(IMG_PATH / hash_objs[j][s]) + ".png"
            trans1_aud = trans1_img.replace('pred_img', 'pred_audio').replace("png", "wav")
            trans2_aud = trans2_img.replace('pred_img', 'pred_audio').replace("png", "wav")
            assert Path(trans1_img).exists(), f"{trans1_img} does not exist!"
            assert Path(trans2_img).exists(), f"{trans2_img} does not exist!"
            assert Path(trans1_aud).exists(), f"{trans1_aud} does not exist!"
            assert Path(trans2_aud).exists(), f"{trans2_aud} does not exist!"

            # create sample_dict to store sample in final_selection
            sample_dict = {}
            sample_dict["audio"] = gt_hash[s][0]
            sample_dict["images"]= [trans1_img, trans2_img]
            sample_dict["audio_trans"] = [trans1_aud, trans2_aud]
            sample_dict["type"] = "SCORE"
            final_selection.append(sample_dict)

        # save the list of sampled questions to a dict
        with open(self.STORE_PATH + "/score_samples.json", "w") as f:
            json.dump(final_selection, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract interesting examples for MIDI/Score domain!")

    # Define arguments
    parser.add_argument('-m', required=True, help="If 1, generate MIDI examples, if 0, generate score examples!")

    args = parser.parse_args()

    if int(args.m):
        ex = ExMidiBasic(model1='hft', model2='oaf')
        ex.gen_examples()
        ex.sample("./data/midi_examples/examples.json", NUM_EXAMPLES=500)
    else:
        ex = ExScore()
        ex.gen_examples()
        ex.sample("data/score_examples/output.xlsx", NUM_EXAMPLES=500)
