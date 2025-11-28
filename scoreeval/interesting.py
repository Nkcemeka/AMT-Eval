"""
    Filename: interesting.py
    Description: This file contains functionality for extracting interesting 
                 examples for user study.
"""

# Import necessary modules
import argparse
import pretty_midi
import itertools
from pathlib import Path
import numpy as np
import sys
sys.path.append("./extras")
import json
import torch
import snap2midi as s2m
from tqdm import tqdm
from utils import trans, get_scores_kong, get_scores_hft, get_scores_oaf, get_scores_trans, beyer_midi_xml
from constants import KONG_CHECKPOINT, KONG_EXT_CONFIG, KONG_PEDAL_CHECKPOINT, HFT_CHECKPOINT, OAF_CHECKPOINT
from metric_utils import xml_to_midi

class IEv1:
    def __init__(self, audioPath: str="./data/audio", midiPath: str = "./data/midi", \
                 midiScorePath: str="./data/midi_score", xmlScorePath: str="./data/xml_score",
                 ONSET_OFFSET_FLAG: bool=False, EXTEND_FLAG: bool=False) -> None:
        """
            Instantiate IEv1

            Args:
            -----
                audioPath (str): Path to directory containing audio files
                midiPath (str): Path to directory containing p-midi files
                midiScorePath (str): Path to directory containing midi score files
                xmlScorePath (str): Path to directory containing musicXML score files
            
            Returns:
            --------
                None
        """
        self.audio_files = sorted(Path(audioPath).glob("*.wav"))[:20]
        self.midi_files = sorted(Path(midiPath).glob("*.mid"))[:20]
        self.midi_score_files = sorted(Path(midiScorePath).glob("*.mid"))[:20]
        self.xml_score_files = sorted(Path(xmlScorePath).glob("*.musicxml"))[:20]

        # Assertion to ensure number of audio, midi, midi_score and xml_score files are the same
        assert len(self.audio_files) == len(self.midi_files) == len(self.midi_score_files) == len(self.xml_score_files), "\
            Number of audio, midi, midi_score and xml_score files are unequal!"
        
        self.inference = s2m.inference.Inference()
        self.ONSET_OFFSET_FLAG = ONSET_OFFSET_FLAG # if false, it uses onset-only F1 score
        self.EXTEND_FLAG = EXTEND_FLAG # if true, extends the note durations when computing scores
        self.STORE_PATH = "/home/nkcemeka/Documents/ismir2026/scoreeval/data/interesting"
    
    def gen_interesting(self):
        interesting_files = [] # List of interesting files
        # Create self.STORE_PATH and necessary subdirectories if they do not exist
        Path(self.STORE_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.STORE_PATH + "/pred_midi").mkdir(parents=True, exist_ok=True) 
        Path(self.STORE_PATH + "/pred_xml").mkdir(parents=True, exist_ok=True) 
        Path(self.STORE_PATH + "/pred_mscore").mkdir(parents=True, exist_ok=True) 

        for audio_file, midi_file, midi_score_file, xml_score_file in tqdm(zip(self.audio_files, \
            self.midi_files, self.midi_score_files, self.xml_score_files), total=len(self.audio_files),\
            desc="Processing files"):

            assert audio_file.stem == midi_file.stem, f"Audio and MIDI file names do not match: {audio_file.stem} != {midi_file.stem}"
            trans_midi = trans(audio_file)

            try:
                # We use the try-except clause for edge cases where measure is short or fast (1s)
                # and some model fails to predict anything for whatever weird reason
                if self.ONSET_OFFSET_FLAG:
                    _, score_trans = get_scores_trans(trans_midi, str(midi_file), extend_flag=self.EXTEND_FLAG)
                else:
                    score_trans, _ = get_scores_trans(trans_midi, str(midi_file), extend_flag=self.EXTEND_FLAG)
            except:
                continue
            torch.cuda.empty_cache()

            # KONG transcription
            kong_midi = self.inference.inference_kong(audio_file, checkpoint_note_path=KONG_CHECKPOINT, \
                                        checkpoint_pedal_path=KONG_PEDAL_CHECKPOINT,\
                                        filename=None, user_ext_config=KONG_EXT_CONFIG)
            
            try:
                if self.ONSET_OFFSET_FLAG:
                    _, score_kong = get_scores_kong(kong_midi, str(midi_file), extend_flag=self.EXTEND_FLAG)
                else:
                    score_kong, _ = get_scores_kong(kong_midi, str(midi_file), extend_flag=self.EXTEND_FLAG)
            except:
                continue
            torch.cuda.empty_cache()

            # OAF transcription
            oaf_midi = self.inference.inference_oaf(audio_file, checkpoint_path=OAF_CHECKPOINT, filename=None)
            try:
                if self.ONSET_OFFSET_FLAG:
                    _, score_oaf = get_scores_oaf(oaf_midi, str(midi_file), extend_flag=self.EXTEND_FLAG)
                else:
                    score_oaf, _ = get_scores_oaf(oaf_midi, str(midi_file), extend_flag=self.EXTEND_FLAG)
            except:
                continue
            torch.cuda.empty_cache()

            # HFT transcription
            hft_midi = self.inference.inference_hft(audio_file, checkpoint_path=HFT_CHECKPOINT, filename=None)
            try:
                if self.ONSET_OFFSET_FLAG:
                    _, score_hft = get_scores_hft(hft_midi, str(midi_file), extend_flag=self.EXTEND_FLAG)
                else:
                    score_hft, _ = get_scores_hft(hft_midi, str(midi_file), extend_flag=self.EXTEND_FLAG)
            except:
                continue
            torch.cuda.empty_cache()

            # Store scores in an arr
            arr_scores = [score_trans, score_kong, score_oaf, score_hft]
            indices = np.argsort(arr_scores)[::-1]

            # Get the location of the top two maximum scores
            # First sort the scores in descending order and use the argsort to get the indices
            top1_index = indices[0]
            top2_index = indices[1]

            if (arr_scores[top1_index] - arr_scores[top2_index]) > 0.1:
                # if the difference between the top two scores is more than 0.1, we consider it an
                # interesting file worth exploring

                # store the predicted midis from the models
                file_stem = Path(audio_file).stem
                trans_path = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_transkun.mid")
                kong_path = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_kong.mid")
                oaf_path = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_oaf.mid")
                hft_path = str(Path(self.STORE_PATH) / f"pred_midi/{file_stem}_hft.mid")
                trans_midi.write(trans_path)
                kong_midi.write(kong_path)
                oaf_midi.write(oaf_path)
                hft_midi.write(hft_path)
                del trans_midi, kong_midi, hft_midi, oaf_midi

                # Predict the MusicXMLs
                trans_path_xml = str(trans_path).replace("pred_midi", "pred_xml").replace("mid", "musicxml")
                kong_path_xml = str(kong_path).replace("pred_midi", "pred_xml").replace("mid", "musicxml")
                oaf_path_xml = str(oaf_path).replace("pred_midi", "pred_xml").replace("mid", "musicxml")
                hft_path_xml = str(hft_path).replace("pred_midi", "pred_xml").replace("mid", "musicxml")

                beyer_midi_xml(trans_path, trans_path_xml)
                beyer_midi_xml(kong_path, kong_path_xml)
                beyer_midi_xml(oaf_path, oaf_path_xml)
                beyer_midi_xml(hft_path, hft_path_xml)

                # convert the predicted xmls to predicted midi scores
                trans_path_mscore = str(trans_path).replace("pred_midi", "pred_mscore")
                kong_path_mscore = str(kong_path).replace("pred_midi", "pred_mscore")
                oaf_path_mscore = str(oaf_path).replace("pred_midi", "pred_mscore")
                hft_path_mscore = str(hft_path).replace("pred_midi", "pred_mscore")

                xml_to_midi(trans_path_xml, trans_path_mscore)
                xml_to_midi(kong_path_xml, kong_path_mscore)
                xml_to_midi(oaf_path_xml, oaf_path_mscore)
                xml_to_midi(hft_path_xml, hft_path_mscore)

                try:
                    # If the conversion fails for some weird reason, skip
                    assert Path(trans_path_mscore).exists()
                    assert Path(kong_path_mscore).exists()
                    assert Path(oaf_path_mscore).exists()
                    assert Path(hft_path_mscore).exists()
                except:
                    continue

                # dict to store info for this interesting segment
                int_dict = {
                    'audio': str(audio_file),
                    'midi': str(midi_file),
                    'midi_score': str(midi_score_file), 
                    'xml_score': str(xml_score_file),
                    'trans': str(trans_path),
                    'kong': str(kong_path),
                    'oaf': str(oaf_path),
                    'hft': str(hft_path),
                    'trans_xml': trans_path_xml,
                    'kong_xml': kong_path_xml,
                    'oaf_xml': oaf_path_xml,
                    'hft_xml': hft_path_xml,
                    'trans_mscore': trans_path_mscore,
                    'oaf_mscore': oaf_path_mscore,
                    'kong_mscore': kong_path_mscore,
                    'hft_mscore': hft_path_mscore
                }
                interesting_files.append(int_dict)
        
        # Create a dict to store in a json file
        json_out = {
            'files': interesting_files
        }

        with open(f"{str(self.STORE_PATH)}/interest.json", "w", encoding="utf-8") as f:
            json.dump(json_out, f, indent=4)

class IEv2:
    def __init__(self):
        raise NotImplemented

    def combinations(self, items: list) -> list[tuple]:
        """
            Finds number of unique pairs from items.

            Args:
            -----
                items (list): List of individual items
            
            Returns:
            --------
                res (list[tuple]): List of unique pairs
        """
        return list(itertools.combinations(items, 2))


ie = IEv1(ONSET_OFFSET_FLAG=True)
ie.gen_interesting()
