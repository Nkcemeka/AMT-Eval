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
import sys
sys.path.append("./extras")
import json
import torch
import snap2midi as s2m
from tqdm import tqdm
from utils import trans, beyer_midi_xml, nakamura_inference, pm2s_inference, generate_piano_roll,\
      musescore_convert, musescore_convert_img, get_note_scores, compute_activation_metrics, \
      compute_mpteval
from constants import KONG_CHECKPOINT, KONG_EXT_CONFIG, KONG_PEDAL_CHECKPOINT, HFT_CHECKPOINT, OAF_CHECKPOINT
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import itertools
import tempfile
import shutil

class ExScore:
    """ 
        Generate examples for the MIDI
        to Score evaluation!
    """
    def __init__(self, audioPath: str="/home/nkcemeka/Documents/ismir2026/scoreeval/data/audio", \
                 midiPath: str = "/home/nkcemeka/Documents/ismir2026/scoreeval/data/midi", \
                 midiScorePath: str="/home/nkcemeka/Documents/ismir2026/scoreeval/data/midi_score", \
                 xmlScorePath: str="/home/nkcemeka/Documents/ismir2026/scoreeval/data/xml_score"):
        
        self.audio_files = sorted(Path(audioPath).glob("*.wav"))[:400]
        self.midi_files = sorted(Path(midiPath).glob("*.mid"))[:400]
        self.midi_score_files = sorted(Path(midiScorePath).glob("*.mid"))[:400]
        self.xml_score_files = sorted(Path(xmlScorePath).glob("*.musicxml"))[:400]
        self.STORE_PATH = "/home/nkcemeka/Documents/ismir2026/scoreeval/data/score_examples"
        
    
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


class ExMidi:
    def __init__(self, audioPath: str="./data/audio", midiPath: str = "./data/midi", \
                 midiScorePath: str="./data/midi_score", xmlScorePath: str="./data/xml_score",
                 ONSET_OFFSET_FLAG: bool=False, REF_THRESH: float=0.85,\
                 REF_SIM_THRESH: float=0.04, MODEl_SIM_THRESH: float=0.1) -> None:
        """
            Instantiate IEMidi

            Args:
                audioPath (str): Path to directory containing audio files
                midiPath (str): Path to directory containing p-midi files
                midiScorePath (str): Path to directory containing midi score files
                xmlScorePath (str): Path to directory containing musicXML score files
                ONSET_OFFSET_FLAG (bool): If true, use F1 (Onset + Offset) instead of Onset only
                REF_THRESH (float): Consider models with scores above REF_THRESH
                REF_SIM_THRESH (float): Reference similarity threshold
                MODEL_SIM_THRESH (float): Model similarity threshold

            Returns:
                None
        """
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
        self.REF_THRESH = REF_THRESH
        self.REF_SIM_THRESH = REF_SIM_THRESH
        self.MODEL_SIM_THRESH = MODEl_SIM_THRESH
    
    def gen_examples(self):
        interesting_files = []
        fab_list = []
        f12_list = []
        idx_dict = defaultdict(dict)

        # Create self.STORE_PATH and necessary subdirectories if they do not exist
        Path(self.STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_midi").mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_proll").mkdir(parents=True, exist_ok=True) 

        for idx, (audio_file, midi_file, midi_score_file, xml_score_file) in tqdm(enumerate(zip(self.audio_files, \
            self.midi_files, self.midi_score_files, self.xml_score_files)), total=len(self.audio_files),\
            desc="Processing files"):

            assert audio_file.stem == midi_file.stem, f"Audio and MIDI file names do not match: {audio_file.stem} != {midi_file.stem}"
            trans_midi = trans(audio_file)
            # Remove sustain from Transkun
            for instrument in trans_midi.instruments:
                instrument.control_changes = [
                    cc for cc in instrument.control_changes if cc.number != 64
                ]
            try:
                # We use the try-except clause for edge cases where measure is short or fast (1s)
                # and some model fails to predict anything for whatever weird reason
                score_trans_dict = get_note_scores(trans_midi, str(midi_file))
                if self.ONSET_OFFSET_FLAG:
                    score_trans = score_trans_dict["f_off"]
                else:
                    score_trans = score_trans_dict["f"]
            except:
                continue
            torch.cuda.empty_cache()

            # KONG transcription
            kong_midi = self.inference.inference_kong(audio_file, checkpoint_note_path=KONG_CHECKPOINT, \
                                        checkpoint_pedal_path=KONG_PEDAL_CHECKPOINT,\
                                        filename=None, user_ext_config=KONG_EXT_CONFIG)
            
            try:
                score_kong_dict = get_note_scores(kong_midi, str(midi_file))
                if self.ONSET_OFFSET_FLAG:
                    score_kong = score_kong_dict["f_off"]
                else:
                    score_kong = score_kong_dict["f"]
            except:
                continue
            torch.cuda.empty_cache()

            # OAF transcription
            oaf_midi = self.inference.inference_oaf(audio_file, checkpoint_path=OAF_CHECKPOINT, filename=None)
            try:
                score_oaf_dict = get_note_scores(oaf_midi, str(midi_file))
                if self.ONSET_OFFSET_FLAG:
                    score_oaf = score_oaf_dict["f_off"]
                else:
                    score_oaf = score_oaf_dict["f"]
            except:
                continue
            torch.cuda.empty_cache()

            # HFT transcription
            hft_midi = self.inference.inference_hft(audio_file, checkpoint_path=HFT_CHECKPOINT, filename=None)
            try:
                score_hft_dict = get_note_scores(hft_midi, str(midi_file))
                if self.ONSET_OFFSET_FLAG:
                    score_hft = score_hft_dict["f_off"]
                else:
                    score_hft = score_hft_dict["f"]
            except:
                continue
            torch.cuda.empty_cache()

            # Store scores in an arr
            arr_scores = [score_trans, score_kong, score_oaf, score_hft]
            arr_scores_midi = [trans_midi, kong_midi, oaf_midi, hft_midi]
            model_map = ['transkun', 'kong', 'oaf', 'hft']
            indices = np.argsort(arr_scores)[::-1]

            # Get the location of the top two maximum scores
            # First sort the scores in descending order and use the argsort to get the indices
            top1_index = indices[0]
            top2_index = indices[1]

            if (arr_scores[top1_index] < self.REF_THRESH) or  (arr_scores[top2_index] < self.REF_THRESH):
                # skip this if it is below the reference threshold
                continue
                
            # find absolute difference in f1 scores between the two models
            fab = abs(arr_scores[top1_index] - arr_scores[top2_index])

            # Get the f1 score between the two models
            try:
                f12 = self.compute_f12(arr_scores_midi[top1_index], arr_scores_midi[top2_index])
                print(f12, audio_file, arr_scores[top1_index], arr_scores[top2_index])
            except:
                continue
                
            fab_list.append(fab)
            f12_list.append(f12)

            # init the dictionary
            for k, each in enumerate([model_map[top1_index], model_map[top2_index]]):
                if k == 0:
                    idx_dict[idx][each] = arr_scores_midi[top1_index]
                else:
                    idx_dict[idx][each] = arr_scores_midi[top2_index]
            
        # Filter based on scatter plot
        # threshold for fr1-fr2 when its less than or equal to REF_SIM_THRESH
        if fab_list and f12_list:
            # Make a scatter plot
            plt.scatter(fab_list, np.abs(1-np.array(f12_list)))
            plt.xlabel("|Fr1 -Fr2|")
            plt.ylabel("|1 - F12|")
            plt.ylim(bottom=0)
            plt.title('Scatter plot of |1 - F12| against |Fr1 -Fr2|')
            plt.savefig(str(Path(self.STORE_PATH)/"fig.png"))

            fab_arr = np.array(fab_list)
            mask1 = fab_arr < self.REF_SIM_THRESH

            # threshold for 1-f12 for values greater than 0.10
            one_minus_f12_arr = np.array(1-np.array(f12_list))
            mask2 = one_minus_f12_arr >= self.MODEL_SIM_THRESH

            # Generate a final mask
            final_mask = mask1 & mask2
            filtered_idx = np.array(list(idx_dict.keys()))[final_mask]
        else:
            filtered_idx = np.array([]) # we do this to avoid bugs
        
        if filtered_idx.size > 0:
            # Save these interesting files
            for each in range(len(filtered_idx)):
                audio_file = self.audio_files[filtered_idx[each].item()]
                midi_file = self.midi_files[filtered_idx[each].item()]
                midi_score_file = self.midi_score_files[filtered_idx[each].item()]
                xml_score_file = self.xml_score_files[filtered_idx[each].item()]

                index = filtered_idx[each]
                temp_dict = idx_dict[index]
                file_stem = Path(audio_file).stem
                # dict to store info for this interesting segment
                int_dict = {
                    'audio': str(audio_file),
                    'midi': str(midi_file),
                    'midi_score': str(midi_score_file), 
                    'xml_score': str(xml_score_file),
                }
                valid = True

                for model_name in temp_dict.keys():
                    # save the pred_midi for model_name
                    midi_path = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_{model_name}.mid")
                    midi_obj = temp_dict[model_name]
                    midi_obj.write(midi_path)

                    # Store the Performance MIDI as a piano roll for the model
                    generate_piano_roll(midi_obj, str(Path(self.STORE_PATH) / f"pred_proll/{file_stem}_{model_name}.png"))

                    # dict to store info for this interesting segment
                    int_dict[f'{model_name}_pmidi'] = midi_path

                if valid:
                    interesting_files.append(int_dict)
        
        # save interesting files
        # Create a dict to store in a json file
        if interesting_files:
            json_out = {
                'files': interesting_files
            }

            with open(f"{str(self.STORE_PATH)}/examples.json", "w", encoding="utf-8") as f:
                json.dump(json_out, f, indent=4)
                
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
    
class ExPre:
    """
        Generates examples for transcription outputs with 
        a good note-level score but (low precision, high recall) and
        (high precision, good recall). In essence, we are looking for
        all possible pairs (A, B) where A has (low precision, good recall)
        and B has (high precision, good recall). We say good/moderate
        recall because a low recall is not tenable. We use the Onset F1
        because we want to investigate places where the pitch and onset is 
        right modt of the time. We do not consider offsets because we want to see what the model
        does next after getting these notes. Does it hallucinate or not?
    """
    def __init__(self, audioPath: str="./data/audio", midiPath: str = "./data/midi", \
                 midiScorePath: str="./data/midi_score", xmlScorePath: str="./data/xml_score",
                 ONSET_OFFSET_FLAG: bool=False, REF_THRESH: float=0.85,\
                 LOW_PREC_THRESH: float=0.40, HIGH_PREC_THRESH: float=0.7, RECALL_THRESH: float=0.7) -> None:
        """
            Instantiate ExPre

            Args:
                audioPath (str): Path to directory containing audio files
                midiPath (str): Path to directory containing p-midi files
                midiScorePath (str): Path to directory containing midi score files
                xmlScorePath (str): Path to directory containing musicXML score files
                ONSET_OFFSET_FLAG (bool): If true, use F1 (Onset + Offset) instead of Onset only
                LOW_PREC_THRESH (float): Low precision threshold
                HIGH_PREC_THRESH (float): High precision threshold
                RECALL_THRESH (float): Recall threshold

            Returns:
                None
        """
        self.audio_files = sorted(Path(audioPath).glob("*.wav"))
        self.midi_files = sorted(Path(midiPath).glob("*.mid"))
        self.midi_score_files = sorted(Path(midiScorePath).glob("*.mid"))
        self.xml_score_files = sorted(Path(xmlScorePath).glob("*.musicxml"))

        # Assertion to ensure number of audio, midi, midi_score and xml_score files are the same
        assert len(self.audio_files) == len(self.midi_files) == len(self.midi_score_files) == len(self.xml_score_files), "\
            Number of audio, midi, midi_score and xml_score files are unequal!"
        
        self.inference = s2m.inference.Inference()
        self.ONSET_OFFSET_FLAG = ONSET_OFFSET_FLAG # if false, it uses onset-only F1 score
        self.STORE_PATH = "/home/nkcemeka/Documents/ismir2026/scoreeval/data/expre_examples"
        self.REF_THRESH = REF_THRESH
        self.LOW_PREC_THRESH = LOW_PREC_THRESH
        self.HIGH_PREC_THRESH = HIGH_PREC_THRESH
        self.RECALL_THRESH = RECALL_THRESH
    
    def gen_examples(self):
        interesting_files = []

        # Create self.STORE_PATH and necessary subdirectories if they do not exist
        Path(self.STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_midi").mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_proll").mkdir(parents=True, exist_ok=True) 

        for idx, (audio_file, midi_file, midi_score_file, xml_score_file) in tqdm(enumerate(zip(self.audio_files, \
            self.midi_files, self.midi_score_files, self.xml_score_files)), total=len(self.audio_files),\
            desc="Processing files"):

            assert audio_file.stem == midi_file.stem, f"Audio and MIDI file names do not match: {audio_file.stem} != {midi_file.stem}"
            scores_dict = {'transkun': {'p': [], 'r': [], 'note_score': [], 'midi': []}, \
                           'kong': {'p': [], 'r': [], 'note_score': [], 'midi': []}, \
                           'hft': {'p': [], 'r': [], 'note_score': [], 'midi': []}, \
                           'oaf': {'p': [], 'r': [], 'note_score': [], 'midi': []}}
            
            trans_midi = trans(audio_file)
            # Remove sustain from Transkun
            for instrument in trans_midi.instruments:
                instrument.control_changes = [
                    cc for cc in instrument.control_changes if cc.number != 64
                ]

            try:
                # We use the try-except clause for edge cases where measure is short or fast (1s)
                # and some model fails to predict anything for whatever weird reason
                score_trans_dict = get_note_scores(trans_midi, str(midi_file))
                if self.ONSET_OFFSET_FLAG:
                    score_trans = score_trans_dict["f_off"]
                else:
                    score_trans = score_trans_dict["f"]

                # Get activation-level metrics
                score_act_trans = compute_activation_metrics(trans_midi, str(midi_file))
                trans_p, trans_r = score_act_trans[0], score_act_trans[1]
                scores_dict['transkun']['p'].append(trans_p)
                scores_dict['transkun']['r'].append(trans_r)
                scores_dict['transkun']['note_score'].append(score_trans)
                scores_dict['transkun']['midi'].append(trans_midi)
            except:
                continue
            torch.cuda.empty_cache()

            # KONG transcription
            kong_midi = self.inference.inference_kong(audio_file, checkpoint_note_path=KONG_CHECKPOINT, \
                                        checkpoint_pedal_path=KONG_PEDAL_CHECKPOINT,\
                                        filename=None, user_ext_config=KONG_EXT_CONFIG)
            
            try:
                score_kong_dict = get_note_scores(kong_midi, str(midi_file))
                if self.ONSET_OFFSET_FLAG:
                    score_kong = score_kong_dict["f_off"]
                else:
                    score_kong = score_kong_dict["f"]
                
                # Get activation-level metrics
                score_act_kong = compute_activation_metrics(kong_midi, str(midi_file))
                kong_p, kong_r = score_act_kong[0], score_act_kong[1]
                scores_dict['kong']['p'].append(kong_p)
                scores_dict['kong']['r'].append(kong_r)
                scores_dict['kong']['note_score'].append(score_kong)
                scores_dict['kong']['midi'].append(kong_midi)
            except:
                continue
            torch.cuda.empty_cache()

            # OAF transcription
            oaf_midi = self.inference.inference_oaf(audio_file, checkpoint_path=OAF_CHECKPOINT, filename=None)
            try:
                score_oaf_dict = get_note_scores(oaf_midi, str(midi_file))
                if self.ONSET_OFFSET_FLAG:
                    score_oaf = score_oaf_dict["f_off"]
                else:
                    score_oaf = score_oaf_dict["f"]
                # Get activation-level metrics
                score_act_oaf = compute_activation_metrics(oaf_midi, str(midi_file))
                oaf_p, oaf_r = score_act_oaf[0], score_act_oaf[1]
                scores_dict['oaf']['p'].append(oaf_p)
                scores_dict['oaf']['r'].append(oaf_r)
                scores_dict['oaf']['note_score'].append(score_oaf)
                scores_dict['oaf']['midi'].append(oaf_midi)
            except:
                continue
            torch.cuda.empty_cache()

            # HFT transcription
            hft_midi = self.inference.inference_hft(audio_file, checkpoint_path=HFT_CHECKPOINT, filename=None)
            try:
                score_hft_dict = get_note_scores(hft_midi, str(midi_file))
                if self.ONSET_OFFSET_FLAG:
                    score_hft = score_hft_dict["f_off"]
                else:
                    score_hft = score_hft_dict["f"]

                # Get activation-level metrics
                score_act_hft = compute_activation_metrics(hft_midi, str(midi_file))
                hft_p, hft_r = score_act_hft[0], score_act_hft[1]
                scores_dict['hft']['p'].append(hft_p)
                scores_dict['hft']['r'].append(hft_r)
                scores_dict['hft']['note_score'].append(score_hft)
                scores_dict['hft']['midi'].append(hft_midi)
            except:
                continue
            torch.cuda.empty_cache()
        
            # For all valid possible combinations, store if we have
            # high prec, good recall and good recall, low precision pair
            valid_models = []
            for key in scores_dict.keys():
                if scores_dict[key]['note_score'][0] > self.REF_THRESH:
                    valid_models.append(key)
            
            if len(valid_models) == 1:
                continue
            
            combs = list(itertools.combinations(valid_models, 2))
            for pair in combs:
                int_dict = {
                    'audio': str(audio_file),
                    'midi': str(midi_file),
                    'midi_score': str(midi_score_file), 
                    'xml_score': str(xml_score_file),
                }
                model_A = pair[0]
                model_B = pair[1]
                file_stem = str(Path(str(midi_file)).stem)

                model_A_p, model_A_r = scores_dict[model_A]['p'][0], scores_dict[model_A]['r'][0]
                model_B_p, model_B_r = scores_dict[model_B]['p'][0], scores_dict[model_B]['r'][0]

                # If any of the two conditions below are fulfilled, store them. They are valid.
                midi_filepath_A = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_{model_A}.mid")
                midi_filepath_B = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_{model_B}.mid")
                proll_filepath_A = str(Path(self.STORE_PATH) / f"pred_proll/{file_stem}_{model_A}.png")
                proll_filepath_B = str(Path(self.STORE_PATH) / f"pred_proll/{file_stem}_{model_B}.png")
                save_flag = False
                
                if model_A_p <= self.LOW_PREC_THRESH and model_B_r >= self.RECALL_THRESH:  
                    save_flag = True
                elif model_B_p <= self.LOW_PREC_THRESH and model_A_r >= self.RECALL_THRESH:
                    save_flag = True
                
                if save_flag:
                    scores_dict[model_A]['midi'][0].write(midi_filepath_A)
                    scores_dict[model_B]['midi'][0].write(midi_filepath_B)
                    generate_piano_roll(scores_dict[model_A]['midi'][0], proll_filepath_A)
                    generate_piano_roll(scores_dict[model_B]['midi'][0], proll_filepath_B)
                    int_dict[f'{model_A}_pmidi'] = midi_filepath_A
                    int_dict[f'{model_B}_pmidi'] = midi_filepath_B
                    interesting_files.append(int_dict)
                

        # save interesting files
        # Create a dict to store in a json file
        if interesting_files:
            json_out = {
                'files': interesting_files
            }

            with open(f"{str(self.STORE_PATH)}/examples.json", "w", encoding="utf-8") as f:
                json.dump(json_out, f, indent=4)      

class ExNote:
    """
        When note onsets are correct, how much do note durations 
        matter for score recoverability? How critical is this? 
        Is the Onset F1 enough? How much reliance do we give to
        the Onset F1 score?

        We pick a range: Onset F1 score > 90% but Onset-Offset F1
        difference between 10% to 20%. This is even drastically 
        important because this should affect the rhythmic structure in
        the score domain.
    """
    def __init__(self, audioPath: str="./data/audio", midiPath: str = "./data/midi", \
                 midiScorePath: str="./data/midi_score", xmlScorePath: str="./data/xml_score",
                 REF_THRESH: float=0.90, LOWER_BOUND: float=0.1, HIGHER_BOUND: float=0.2) -> None:
        """
            Instantiate ExNote

            Args:
                audioPath (str): Path to directory containing audio files
                midiPath (str): Path to directory containing p-midi files
                midiScorePath (str): Path to directory containing midi score files
                xmlScorePath (str): Path to directory containing musicXML score files
                REF_THRESH (float): Consider models with scores above REF_THRESH
                LOWER_BOUND (float): Lower bound
                HIGHER_BOUND (float): Higher bound

            Returns:
                None
        """
        self.audio_files = sorted(Path(audioPath).glob("*.wav"))
        self.midi_files = sorted(Path(midiPath).glob("*.mid"))
        self.midi_score_files = sorted(Path(midiScorePath).glob("*.mid"))
        self.xml_score_files = sorted(Path(xmlScorePath).glob("*.musicxml"))

        # Assertion to ensure number of audio, midi, midi_score and xml_score files are the same
        assert len(self.audio_files) == len(self.midi_files) == len(self.midi_score_files) == len(self.xml_score_files), "\
            Number of audio, midi, midi_score and xml_score files are unequal!"
        
        self.inference = s2m.inference.Inference()
        self.STORE_PATH = "/home/nkcemeka/Documents/ismir2026/scoreeval/data/exnote_examples"
        self.REF_THRESH = REF_THRESH
        self.LOWER_BOUND = LOWER_BOUND
        self.HIGHER_BOUND = HIGHER_BOUND

    def gen_examples(self):
        interesting_files = []

        # Create self.STORE_PATH and necessary subdirectories if they do not exist
        Path(self.STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_midi").mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_proll").mkdir(parents=True, exist_ok=True) 

        for idx, (audio_file, midi_file, midi_score_file, xml_score_file) in tqdm(enumerate(zip(self.audio_files, \
            self.midi_files, self.midi_score_files, self.xml_score_files)), total=len(self.audio_files),\
            desc="Processing files"):

            assert audio_file.stem == midi_file.stem, f"Audio and MIDI file names do not match: {audio_file.stem} != {midi_file.stem}"
            scores_dict = {'transkun': {'f': [], 'f_off': [], 'midi': []}, \
                           'kong': {'f': [], 'f_off': [], 'midi': []}, \
                           'hft': {'f': [], 'f_off': [], 'midi': []}, \
                           'oaf': {'f': [], 'f_off': [], 'midi': []}}
            trans_midi = trans(audio_file)

            # Remove sustain from Transkun
            for instrument in trans_midi.instruments:
                instrument.control_changes = [
                    cc for cc in instrument.control_changes if cc.number != 64
                ]

            try:
                # We use the try-except clause for edge cases where measure is short or fast (1s)
                # and some model fails to predict anything for whatever weird reason
                score_trans_dict = get_note_scores(trans_midi, str(midi_file))
                scores_dict['transkun']['f'].append(score_trans_dict['f'])
                scores_dict['transkun']['f_off'].append(score_trans_dict['f_off'])
                scores_dict['transkun']['midi'].append(trans_midi)
            except:
                continue
            torch.cuda.empty_cache()

            # KONG transcription
            kong_midi = self.inference.inference_kong(audio_file, checkpoint_note_path=KONG_CHECKPOINT, \
                                        checkpoint_pedal_path=KONG_PEDAL_CHECKPOINT,\
                                        filename=None, user_ext_config=KONG_EXT_CONFIG)
            
            try:
                score_kong_dict = get_note_scores(kong_midi, str(midi_file))
                scores_dict['kong']['f'].append(score_kong_dict['f'])
                scores_dict['kong']['f_off'].append(score_kong_dict['f_off'])
                scores_dict['kong']['midi'].append(kong_midi)
            except:
                continue
            torch.cuda.empty_cache()

            # OAF transcription
            oaf_midi = self.inference.inference_oaf(audio_file, checkpoint_path=OAF_CHECKPOINT, filename=None)
            try:
                score_oaf_dict = get_note_scores(oaf_midi, str(midi_file))
                scores_dict['oaf']['f'].append(score_oaf_dict['f'])
                scores_dict['oaf']['f_off'].append(score_oaf_dict['f_off'])
                scores_dict['oaf']['midi'].append(oaf_midi)
            except:
                continue
            torch.cuda.empty_cache()

            # HFT transcription
            hft_midi = self.inference.inference_hft(audio_file, checkpoint_path=HFT_CHECKPOINT, filename=None)
            try:
                score_hft_dict = get_note_scores(hft_midi, str(midi_file))
                scores_dict['hft']['f'].append(score_hft_dict['f'])
                scores_dict['hft']['f_off'].append(score_hft_dict['f_off'])
                scores_dict['hft']['midi'].append(hft_midi)
            except:
                continue
            torch.cuda.empty_cache()
        
            # For all valid possible combinations, store if we have
            # high prec, good recall and good recall, low precision pair
            valid_models = []
            for key in scores_dict.keys():
                if scores_dict[key]['f'][0] > self.REF_THRESH:
                    valid_models.append(key)
            
            if len(valid_models) == 1:
                continue
            
            combs = list(itertools.combinations(valid_models, 2))
            for pair in combs:
                int_dict = {
                    'audio': str(audio_file),
                    'midi': str(midi_file),
                    'midi_score': str(midi_score_file), 
                    'xml_score': str(xml_score_file),
                }
                model_A = pair[0]
                model_B = pair[1]
                file_stem = str(Path(str(midi_file)).stem)

                model_A_score = scores_dict[model_A]['f_off'][0]
                model_B_score = scores_dict[model_B]['f_off'][0]
                diff = abs(model_A_score - model_B_score)
                save_flag = False

                if self.LOWER_BOUND <= diff and diff <= self.HIGHER_BOUND:
                    save_flag = True

                # If any of the two conditions below are fulfilled, store them. They are valid.
                midi_filepath_A = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_{model_A}.mid")
                midi_filepath_B = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_{model_B}.mid")
                proll_filepath_A = str(Path(self.STORE_PATH) / f"pred_proll/{file_stem}_{model_A}.png")
                proll_filepath_B = str(Path(self.STORE_PATH) / f"pred_proll/{file_stem}_{model_B}.png")           
                
                if save_flag:
                    scores_dict[model_A]['midi'][0].write(midi_filepath_A)
                    scores_dict[model_B]['midi'][0].write(midi_filepath_B)
                    generate_piano_roll(scores_dict[model_A]['midi'][0], proll_filepath_A)
                    generate_piano_roll(scores_dict[model_B]['midi'][0], proll_filepath_B)
                    int_dict[f'{model_A}_pmidi'] = midi_filepath_A
                    int_dict[f'{model_B}_pmidi'] = midi_filepath_B
                    interesting_files.append(int_dict)            

        # save interesting files
        # Create a dict to store in a json file
        if interesting_files:
            json_out = {
                'files': interesting_files
            }

            with open(f"{str(self.STORE_PATH)}/examples.json", "w", encoding="utf-8") as f:
                json.dump(json_out, f, indent=4)

class ExHarm:
    """
        Cloud Momentum and Cloud Diameter are proposed
        transcription metrics for expressiveness. Cloud 
        Diameter measures tonal dispersion between notes
        in a segment and cloud momentum measures harmonic 
        changes. E.g for example, if the song is approaching
        a climax, it is expected there will be rapid harmonic
        changes which correlate with tempo, dynamics etc.

        However, these metrics should still be able to give clues
        of the veridical aspect of the MIDI file. In essence,
        even without tempo and dynamics, the cloud momentum 
        and diameter should give some insight into harmony. 

        Here, we generate examples with high onset-offset F1 score. 
        An interesting pair will be one with high onset-F1 score and
        drastically different harmonic tendencies. However, we use 
        a range based approach. 

        Onset-offset F1 > 90% but difference in correlation is 10%??
        Rough heuristics...
    """
    def __init__(self, audioPath: str="./data/audio", midiPath: str = "./data/midi", \
                 midiScorePath: str="./data/midi_score", xmlScorePath: str="./data/xml_score",
                 REF_THRESH: float=0.90, DIFF_THRESH: float=0.1) -> None:
        """
            Instantiate ExHarm

            Args:
                audioPath (str): Path to directory containing audio files
                midiPath (str): Path to directory containing p-midi files
                midiScorePath (str): Path to directory containing midi score files
                xmlScorePath (str): Path to directory containing musicXML score files
                REF_THRESH (float): Consider models with Onset-Offset F1 scores above REF_THRESH
                DIFF_THRESH (float): Difference threshold (Default: 0.1)

            Returns:
                None
        """
        self.audio_files = sorted(Path(audioPath).glob("*.wav"))
        self.midi_files = sorted(Path(midiPath).glob("*.mid"))
        self.midi_score_files = sorted(Path(midiScorePath).glob("*.mid"))
        self.xml_score_files = sorted(Path(xmlScorePath).glob("*.musicxml"))

        # Assertion to ensure number of audio, midi, midi_score and xml_score files are the same
        assert len(self.audio_files) == len(self.midi_files) == len(self.midi_score_files) == len(self.xml_score_files), "\
            Number of audio, midi, midi_score and xml_score files are unequal!"
        
        self.inference = s2m.inference.Inference()
        self.STORE_PATH = "/home/nkcemeka/Documents/ismir2026/scoreeval/data/exharm_examples"
        self.REF_THRESH = REF_THRESH
        self.DIFF_THRESH = DIFF_THRESH
    
    def gen_examples(self):
        interesting_files = []

        # Create self.STORE_PATH and necessary subdirectories if they do not exist
        Path(self.STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_midi").mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_proll").mkdir(parents=True, exist_ok=True) 

        for idx, (audio_file, midi_file, midi_score_file, xml_score_file) in tqdm(enumerate(zip(self.audio_files, \
            self.midi_files, self.midi_score_files, self.xml_score_files)), total=len(self.audio_files),\
            desc="Processing files"):

            assert audio_file.stem == midi_file.stem, f"Audio and MIDI file names do not match: {audio_file.stem} != {midi_file.stem}"
            scores_dict = {'transkun': {'f_off': [], 'midi': [], 'cloud_diameter_corr': [], 'cloud_momentum_corr': [],\
                                        'tensile_strain_corr': []}, \
            'kong': {'f_off': [], 'midi': [], 'cloud_diameter_corr': [], 'cloud_momentum_corr': [], 'tensile_strain_corr': []}, \
            'hft': {'f_off': [], 'midi': [], 'cloud_diameter_corr': [], 'cloud_momentum_corr': [], 'tensile_strain_corr': []}, \
            'oaf': {'f_off': [], 'midi': [], 'cloud_diameter_corr': [], 'cloud_momentum_corr': [], 'tensile_strain_corr': []}}
            
            trans_midi = trans(audio_file)

            # Remove sustain from Transkun
            for instrument in trans_midi.instruments:
                instrument.control_changes = [
                    cc for cc in instrument.control_changes if cc.number != 64
                ]

            try:
                # We use the try-except clause for edge cases where measure is short or fast (1s)
                # and some model fails to predict anything for whatever weird reason
                score_trans_dict = get_note_scores(trans_midi, str(midi_file))

                # save trans_midi temporarily
                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.copy(str(midi_file), f"{tmpdir}/gt.mid")
                    trans_midi.write(f"{tmpdir}/pred.mid")
                    trans_mpt_dict = compute_mpteval(f"{tmpdir}/pred.mid", f"{tmpdir}/gt.mid")

                scores_dict['transkun']['f_off'].append(score_trans_dict['f_off'])
                scores_dict['transkun']['midi'].append(trans_midi)
                scores_dict['transkun']['cloud_diameter_corr'].append(trans_mpt_dict['cloud_diameter_corr'])
                scores_dict['transkun']['cloud_momentum_corr'].append(trans_mpt_dict['cloud_momentum_corr'])
                scores_dict['transkun']['tensile_strain_corr'].append(trans_mpt_dict['tensile_strain_corr'])
            except:
                continue
            torch.cuda.empty_cache()

            # KONG transcription
            kong_midi = self.inference.inference_kong(audio_file, checkpoint_note_path=KONG_CHECKPOINT, \
                                        checkpoint_pedal_path=KONG_PEDAL_CHECKPOINT,\
                                        filename=None, user_ext_config=KONG_EXT_CONFIG)
            
            try:
                score_kong_dict = get_note_scores(kong_midi, str(midi_file))

                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.copy(str(midi_file), f"{tmpdir}/gt.mid")
                    kong_midi.write(f"{tmpdir}/pred.mid")
                    kong_mpt_dict = compute_mpteval(f"{tmpdir}/pred.mid", f"{tmpdir}/gt.mid")

                scores_dict['kong']['f_off'].append(score_kong_dict['f_off'])
                scores_dict['kong']['midi'].append(kong_midi)
                scores_dict['kong']['cloud_diameter_corr'].append(kong_mpt_dict['cloud_diameter_corr'])
                scores_dict['kong']['cloud_momentum_corr'].append(kong_mpt_dict['cloud_momentum_corr'])
                scores_dict['kong']['tensile_strain_corr'].append(kong_mpt_dict['tensile_strain_corr'])
            except:
                continue
            torch.cuda.empty_cache()

            # OAF transcription
            oaf_midi = self.inference.inference_oaf(audio_file, checkpoint_path=OAF_CHECKPOINT, filename=None)
            try:
                score_oaf_dict = get_note_scores(oaf_midi, str(midi_file))

                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.copy(str(midi_file), f"{tmpdir}/gt.mid")
                    oaf_midi.write(f"{tmpdir}/pred.mid")
                    oaf_mpt_dict = compute_mpteval(f"{tmpdir}/pred.mid", f"{tmpdir}/gt.mid")

                scores_dict['oaf']['f_off'].append(score_oaf_dict['f_off'])
                scores_dict['oaf']['midi'].append(oaf_midi)
                scores_dict['oaf']['cloud_diameter_corr'].append(oaf_mpt_dict['cloud_diameter_corr'])
                scores_dict['oaf']['cloud_momentum_corr'].append(oaf_mpt_dict['cloud_momentum_corr'])
                scores_dict['oaf']['tensile_strain_corr'].append(oaf_mpt_dict['tensile_strain_corr'])
            except:
                continue
            torch.cuda.empty_cache()

            # HFT transcription
            hft_midi = self.inference.inference_hft(audio_file, checkpoint_path=HFT_CHECKPOINT, filename=None)
            try:
                score_hft_dict = get_note_scores(hft_midi, str(midi_file))

                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.copy(str(midi_file), f"{tmpdir}/gt.mid")
                    hft_midi.write(f"{tmpdir}/pred.mid")
                    hft_mpt_dict = compute_mpteval(f"{tmpdir}/pred.mid", f"{tmpdir}/gt.mid")

                scores_dict['hft']['f_off'].append(score_hft_dict['f_off'])
                scores_dict['hft']['midi'].append(hft_midi)
                scores_dict['hft']['cloud_diameter_corr'].append(hft_mpt_dict['cloud_diameter_corr'])
                scores_dict['hft']['cloud_momentum_corr'].append(hft_mpt_dict['cloud_momentum_corr'])
                scores_dict['hft']['tensile_strain_corr'].append(hft_mpt_dict['tensile_strain_corr'])
            except:
                continue
            torch.cuda.empty_cache()
        
            # For all valid possible combinations, store if we have
            # high prec, good recall and good recall, low precision pair
            valid_models = []
            for key in scores_dict.keys():
                if scores_dict[key]['f_off'][0] > self.REF_THRESH:
                    valid_models.append(key)
            
            if len(valid_models) == 1:
                continue
            
            combs = list(itertools.combinations(valid_models, 2))
            for pair in combs:
                int_dict = {
                    'audio': str(audio_file),
                    'midi': str(midi_file),
                    'midi_score': str(midi_score_file), 
                    'xml_score': str(xml_score_file)
                }
                model_A = pair[0]
                model_B = pair[1]
                file_stem = str(Path(str(midi_file)).stem)

                model_A_cd = scores_dict[model_A]['cloud_diameter_corr'][0]
                model_B_cd = scores_dict[model_B]['cloud_diameter_corr'][0]
                model_A_cm = scores_dict[model_A]['cloud_momentum_corr'][0]
                model_B_cm = scores_dict[model_B]['cloud_momentum_corr'][0]
                model_A_ts = scores_dict[model_A]['tensile_strain_corr'][0]
                model_B_ts = scores_dict[model_B]['tensile_strain_corr'][0]

                save_flag = False
                metrics_int = [] # interesting harmony metrics to take note of

                if abs(model_A_cd - model_B_cd) >= self.DIFF_THRESH:
                    save_flag = True
                    metrics_int.append("cloud_diameter_corr")
                
                if abs(model_A_cm - model_B_cm) >= self.DIFF_THRESH:
                    save_flag = True
                    metrics_int.append("cloud_momentum_corr")
                
                if abs(model_A_ts - model_B_ts) >= self.DIFF_THRESH:
                    save_flag = True
                    metrics_int.append('tensile_strain-corr')

                # If any of the two conditions below are fulfilled, store them. They are valid.
                midi_filepath_A = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_{model_A}.mid")
                midi_filepath_B = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_{model_B}.mid")
                proll_filepath_A = str(Path(self.STORE_PATH) / f"pred_proll/{file_stem}_{model_A}.png")
                proll_filepath_B = str(Path(self.STORE_PATH) / f"pred_proll/{file_stem}_{model_B}.png")           
                
                if save_flag:
                    scores_dict[model_A]['midi'][0].write(midi_filepath_A)
                    scores_dict[model_B]['midi'][0].write(midi_filepath_B)
                    generate_piano_roll(scores_dict[model_A]['midi'][0], proll_filepath_A)
                    generate_piano_roll(scores_dict[model_B]['midi'][0], proll_filepath_B)
                    int_dict[f'{model_A}_pmidi'] = midi_filepath_A
                    int_dict[f'{model_B}_pmidi'] = midi_filepath_B
                    int_dict[f'metrics_int'] = metrics_int
                    interesting_files.append(int_dict)            

        # save interesting files
        # Create a dict to store in a json file
        if interesting_files:
            json_out = {
                'files': interesting_files
            }

            with open(f"{str(self.STORE_PATH)}/examples.json", "w", encoding="utf-8") as f:
                json.dump(json_out, f, indent=4)


a = ExHarm()
a.gen_examples()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Extract interesting examples for MIDI/Score domain!")

#     # Define arguments
#     parser.add_argument('-m', required=True, help="If 1, generate MIDI examples, if 0, generate score examples!")

#     args = parser.parse_args()
#     if int(args.m):
#         ex = ExMidi(ONSET_OFFSET_FLAG=False, REF_SIM_THRESH=0.07, REF_THRESH=0.9)
#     else:
#         ex = ExScore()
    
#     ex.gen_examples()
