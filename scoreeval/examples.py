"""
    Filename: examples.py
    Description: This file contains functionality for extracting interesting 
                 examples for user study.
"""

# Import necessary modules
import argparse
import pretty_midi
from utils import merge_score, postprocess_score
from pathlib import Path
import numpy as np
import sys
sys.path.append("./extras")
import json
import torch
import snap2midi as s2m
from tqdm import tqdm
from utils import trans, beyer_midi_xml, nakamura_inference, pm2s_inference, generate_piano_roll,\
      musescore_convert, musescore_convert_img, get_note_scores
from constants import KONG_CHECKPOINT, KONG_EXT_CONFIG, KONG_PEDAL_CHECKPOINT, HFT_CHECKPOINT, OAF_CHECKPOINT
from collections import defaultdict
import matplotlib.pyplot as plt
import os

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
                postprocess_score(pm2s_xml_path, pm2s_xml_path)

                # merge nakamura before conversion to xml
                musescore_convert(nak_mscore_path, nak_xml_path)
                merge_score(nak_xml_path, nak_xml_path)
                # For nakamura, strip the final score off the tempo information for rendering
                # purposes
                postprocess_score(nak_xml_path, nak_xml_path)

                # For musescore, get the xml first
                musescore_convert(str(midi_file), musescore_xml_path)
                # now for musescore, get the mscore
                musescore_convert(musescore_xml_path, musescore_mscore_path)
                # now strip the xml of the tempo information for rendering processes
                postprocess_score(musescore_xml_path, musescore_xml_path)

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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract interesting examples for MIDI/Score domain!")

    # Define arguments
    parser.add_argument('-m', required=True, help="If 1, generate MIDI examples, if 0, generate score examples!")

    args = parser.parse_args()
    if int(args.m):
        ex = ExMidi(ONSET_OFFSET_FLAG=False, REF_SIM_THRESH=0.07, REF_THRESH=0.9)
    else:
        ex = ExScore()
    
    ex.gen_examples()
