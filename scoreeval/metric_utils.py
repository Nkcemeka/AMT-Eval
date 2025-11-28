from score_transformer import score_similarity
import tempfile
from pathlib import Path
import torch
import shutil
import os
import subprocess
from typing import Optional
from jiwer import wer
import jiwer
from muster import muster
from constants import LD_PATH, MUSESCORE_PATH, MV2H_PATH
from tokenizer import MultistreamTokenizer

# Credits: https://github.com/TimFelixBeyer/MIDI2ScoreTransformer
def score_similarity_normalized(est, gt, full=False):
    if est is None or gt is None:
        return {
            "Clef": None,
            "KeySignature": None,
            "TimeSignature": None,
            "NoteDeletion": None,
            "NoteInsertion": None,
            "NoteSpelling": None,
            "NoteDuration": None,
            "StemDirection": None,
            "Beams": None,
            "Tie": None,
            "StaffAssignment": None,
            "Voice": None,
        }
    sim = score_similarity(est, gt, full=full)
    new_sim = {}
    for k, v in sim.items():
        if v is None:
            new_sim[k] = None
        elif k == "n_Note" or any(key in k for key in ["F1", "Rec", "Prec", "TP", "FP", "FN", "TN"]):
            new_sim[k] = v
        else:
            new_sim[k] = v / sim["n_Note"]
    return new_sim

# Calculate scoreSimilarity metric and return a dict
def scoreSim(gt_xml: str, pred_xml) -> dict:
    # instantiate our result dict
    metrics_dict_scoresim = {'e_miss': 0, 'e_extra': 0, 'e_dur': 0,\
        'e_staff': 0, 'e_stem': 0, 'e_spell': 0}

    # Calculate the score similarity metrics
    sim = score_similarity_normalized(pred_xml, gt_xml, full=False)

    # update our result dict
    metrics_dict_scoresim['e_miss'] += sim['NoteDeletion']
    metrics_dict_scoresim['e_extra'] += sim['NoteInsertion']
    metrics_dict_scoresim['e_dur'] += sim['NoteDuration']
    metrics_dict_scoresim['e_staff'] += sim['StaffAssignment']
    metrics_dict_scoresim['e_stem'] += sim['StemDirection']
    metrics_dict_scoresim['e_spell'] += sim['NoteSpelling']

    return metrics_dict_scoresim

def scoreMuster(gt_xml: str, pred_xml) -> dict:
    """
        Calculates the scores for one pair of files.
    """
    # instaniate our result dict
    metrics_dict_muster = {'e_p': 0, 'e_miss': 0, 'e_extra': 0, 'e_onset': 0, 'e_offset': 0}
    
    # Get the muster metric score
    muster_scores = muster(pred_xml, gt_xml)

    # Return our result dict
    for key in muster_scores.keys():
        if muster_scores[key] is None:
            return None
        
    metrics_dict_muster['e_p'] = muster_scores['PitchER']
    metrics_dict_muster['e_miss'] = muster_scores['MissRate']
    metrics_dict_muster['e_extra'] = muster_scores['ExtraRate']
    metrics_dict_muster['e_onset'] = muster_scores['OnsetER']
    metrics_dict_muster['e_offset'] = muster_scores['OffsetER']

    return metrics_dict_muster


# def xml_to_midi_notemp(xml_path: str, out_path: str) -> str:
#     """
#         Converts MusicXML to MIDI 
#         and stores it in the current working
#         directory!

#         Args:
#         ------
#             xml_path (str): path to musicXML file

#         Returns:
#         --------
#             out_path (str): Returns the path to the MIDI file
#      """
#     # Update Environment Variables
#     evs = os.environ.copy()
#     evs["DISPLAY"] = ":0"
#     evs["QT_QPA_PLATFORM"] = "offscreen"
#     evs["XDG_RUNTIME_DIR"] = tmpdir
#     evs["LD_LIBRARY_PATH"] = LD_PATH + evs.get("LD_LIBRARY_PATH", "")

#     # perform conversion in a new environment
#     subprocess.run(
#         [MUSESCORE_PATH, "-o", out_path, f"{xml_path}"],
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#         env=evs
#     )

#     return out_path

# Convert XML to MIDI using Musescore
def xml_to_midi(xml_path: str, out_path: str) -> str:
    """
        Converts MusicXML to MIDI 
        and stores it!

        Args:
        ------
            xml_path (str): path to musicXML file

        Returns:
        --------
            out_path (str): Returns the path to the MIDI file
     """
    # if out_path is None:
    #     out_path = "./" + str(Path(xml_path).stem) + ".mid"

    # Create temporary directory for the conversion
    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = str(Path(xml_path).suffix)[1:] # we start from 1 to remove the dot
        shutil.copy(xml_path, f"{tmpdir}/temp.{suffix}")

        # Update Environment Variables
        evs = os.environ.copy()
        evs["DISPLAY"] = ":0"
        evs["QT_QPA_PLATFORM"] = "offscreen"
        evs["XDG_RUNTIME_DIR"] = tmpdir
        evs["LD_LIBRARY_PATH"] = LD_PATH + evs.get("LD_LIBRARY_PATH", "")

        # perform conversion in a new environment
        subprocess.run(
            [MUSESCORE_PATH, "-o", out_path, f"{tmpdir}/temp.{suffix}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=evs
        )

    return out_path


# Takes ground truth MIDI file and transcription,
# converts it to txt and then returns the MV2H evaluation
def mv2h_eval(gt_midi: str, pred_midi: str) -> dict:
    """
        This takes two MIDI files and returns the
        MV2H metrics.

        Args:
        -----
            gt_midi (str): Ground truth MIDI score
            pred_midi (str): Predicted MIDI score
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        suffix_gt = str(Path(gt_midi).suffix)[1:]
        suffix_pred = str(Path(pred_midi).suffix)[1:]

        shutil.copy(gt_midi, f"{tmpdir}/gt.{suffix_gt}")
        shutil.copy(pred_midi, f"{tmpdir}/pred.{suffix_pred}")

        # Convert the MIDI files to txt
        cmd_gt_txt = ["java", "-cp", "bin", "mv2h.tools.Converter"\
                ,"-i", f"{tmpdir}/gt.{suffix_gt}", "-o", f"{tmpdir}/gt.txt"]
        cmd_pred_txt = ["java", "-cp", "bin", "mv2h.tools.Converter"\
                ,"-i", f"{tmpdir}/pred.{suffix_pred}", "-o", f"{tmpdir}/pred.txt"]

        subprocess.run(cmd_gt_txt, cwd=MV2H_PATH)
        subprocess.run(cmd_pred_txt, cwd=MV2H_PATH)

        # Now perform alignment process and compute metric
        cmd_mv2h = ["java", "-cp", "bin", "mv2h.Main", "-g", f"{tmpdir}/gt.txt", "-t", \
                f"{tmpdir}/pred.txt", "-a"]
        res = subprocess.run(cmd_mv2h, cwd=MV2H_PATH, check=True, capture_output=True,\
                text=True)
        res = res.stdout
        res_list = res.split("\n")
        if res_list[-1] == '':
            res_list.pop()
        res_dict = {}
        for each in res_list[-6:]:
            key, value = each.split(': ')
            value = value.strip()
            res_dict[key] = float(value)
        return res_dict

def get_seq(tokens: torch.Tensor) -> list:
    TOKEN_NAME_MAP = {
        'offset': 'mo',
        'downbeat': 'db',
        'duration': 'md',
        'grace': 'g',
        'pitch': 'p',
        'accidental': 'a',
        'keysignature': 'k',
        'velocity': 'v',
        'trill': 't',
        'staccato': 'st',
        'voice': 'vo',
        'stem': 'sd',
        'hand': 'h'
    }
    stream_indices = {}
    for key, value in tokens.items():
        idx = torch.argmax(value, dim=-1) # size N
        stream_indices[key] = idx

    # Create composite tokens for each note
    seq = []
    for i in range(tokens['offset'].shape[0]):
        parts = []
        for key in tokens.keys():
            if key not in ['offset', 'duration', 'pitch']:
                continue
            prefix = TOKEN_NAME_MAP[key]
            parts.append(f"{prefix}{int(stream_indices[key][i])}")
        token = '_'.join(parts)
        seq.append(token)
    return seq

def compute_ed_metrics(y_true, y_pred):
    def levenshtein(a, b):
        n, m = len(a), len(b)

        if n > m:
            a, b = b, a
            n, m = m, n

        current = range(n + 1)
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]

    ed_acc = 0
    length_acc = 0
    label_acc = 0
    for t, h in zip(y_true, y_pred):
        ed = levenshtein(t, h)
        ed_acc += ed
        length_acc += len(t)
        if ed > 0:
            label_acc += 1

    return {
        "sym-er": 100.0 * ed_acc / length_acc,
        "seq-er": 100.0 * label_acc / len(y_pred),
    }

def scoreSer(gt_xml: str, pred_xml: str) -> float:
    """
        Calculates the symbolic error rate!

        Args:
        -----
            gt_xml (str): Ground truth musicXML file
            pred_xml (str): Predicted musicXML file


        Returns:
        ---------
            result (dict): Dict containing symoblic and
                           sequence error rate.
    """
    gt = MultistreamTokenizer().tokenize_mxl(gt_xml)
    pred = MultistreamTokenizer().tokenize_mxl(pred_xml)

    # Get sequences
    seq_gt = get_seq(gt)
    seq_pred = get_seq(pred)
    #print(seq_gt, seq_pred)
    #print(wer(' '.join(seq_gt), ' '.join(seq_pred)))
    return compute_ed_metrics([seq_gt], [seq_pred])

