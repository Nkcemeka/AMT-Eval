from typing import Optional
import moduleconf
import torch
import pretty_midi
import numpy as np
from mir_eval.transcription import precision_recall_f1_overlap as prf
from externals.Transkun.transkun.transcribe import readAudio, writeMidi
from externals.MIDI2ScoreTransformer.midi2scoretransformer.utils import infer
from externals.MIDI2ScoreTransformer.midi2scoretransformer.tokenizer import MultistreamTokenizer
from externals.MIDI2ScoreTransformer.midi2scoretransformer.models.roformer import Roformer
from externals.MIDI2ScoreTransformer.midi2scoretransformer.score_utils import postprocess_score
from externals.PM2S.pm2s import CRNNJointPM2S
import subprocess
from pathlib import Path
import tempfile
import music21 as m21
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
from constants import MUSESCORE_PATH, LD_PATH
import os
from typing import Union, Iterable
from PIL import Image
from PIL.ImageFile import ImageFile
import glob
from peamt import PEAMT
import mpteval
import partitura as pt
from collections import defaultdict
from constants import LD_PATH, MUSESCORE_PATH, MV2H_PATH
from score_transformer import score_similarity
from muster import muster
from bs4 import BeautifulSoup
from bs4.element import Tag
import shutil
import uuid
import zipfile
from fractions import Fraction
from copy import deepcopy

# Define PathLike
PathLike = Union[str, bytes, os.PathLike, Path]


# ============================
# Section 1: MIDI Utilities
# ============================

def midi_to_csv(midi_obj: pretty_midi.PrettyMIDI):
    """
        Convert a pretty MIDI object to a pandas DataFrame and save as csv

        Args:
            midi_obj (pretty_midi.PrettyMIDI): PrettyMIDI object
    """
    note_events = []
    for note in midi_obj.instruments[0].notes:
        note_events.append([note.start, note.end-note.start, note.pitch, note.velocity/128, 1])
    df = pd.DataFrame(note_events, columns=['start', 'duration', 'pitch', 'velocity', 'label'])
    return df

def csv_to_list(csv):
    """
    Convert a csv score file to a list of note events
    Credits: `https://link.springer.com/book/10.1007/978-3-030-69808-9`

    Args:
        csv (str or pd.DataFrame): Either a path to a csv file or a data frame

    Returns:
        score (list): A list of note events where each note is specified as
            ``[start, duration, pitch, velocity, label]``
    """

    if isinstance(csv, str):
        df = pd.read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        df = csv
    else:
        raise RuntimeError('csv must be a path to a csv file or pd.DataFrame')

    score = []
    for i, (start, duration, pitch, velocity, label) in df.iterrows():
        score.append([start, duration, pitch, velocity, label])
    return score


# ====================================
# Section 2: Visulalization Utilities
# ====================================

def draw_keyboard(ax, pitch_min, pitch_max):
    for midi_num in range(int(pitch_min), int(pitch_max)+1):
        # Determine if it's a black or white key
        note_name = (midi_num % 12)
        is_black = note_name in [1,3,6,8,10]  # C#, D#, F#, G#, A#
        color = 'black' if is_black else 'white'
        rect = patches.Rectangle((0, midi_num-0.5), 1, 1, facecolor=color, edgecolor='k')
        ax.add_patch(rect)

        if not is_black:
            ax.text(0.5, midi_num, pretty_midi.utilities.note_number_to_name(midi_num),
                    ha='center', va='center',
                    fontsize=8, color='black')

    ax.set_xlim(0, 1)
    ax.set_ylim(pitch_min-1.5, pitch_max+1.5)
    ax.axis('off')

def visualize_piano_roll(score: list, output_path: str, xlabel: str='Time (seconds)', ylabel:str='', \
                         colors: str='viridis', velocity_alpha: bool=False,
                         figsize: tuple=(16, 8), ax=None, dpi: int=300, save=True):
    """
        Plot a pianoroll visualization

        Args:
            score (list): List of note events
            output_path (str): Path to save the piano roll visualization
            xlabel (str): Label for x axis (Default value = 'Time (seconds)')
            ylabel (str): Label for y axis (Default value = 'Pitch' or '')
            colors (str): Several options: 1. string of matplotlib colormap,
                2. list or np.ndarray of matplotlib color specifications
            velocity_alpha (bool): Use the velocity value for the alpha value of the corresponding rectangle
                (Default value = False)
            figsize (tuple): Width, height in inches (Default value = (12)
            ax: The Axes instance to plot on (Default value = None)
            dpi: int: Dots per inch (Default value = 300)

        Returns:
            fig: The created matplotlib figure or None if ax was given.
            ax: The used axes
    """
    # fig = None
    # if ax is None:
    #     fig = plt.figure(figsize=figsize, dpi=dpi)
    #     ax = plt.subplot(1, 1, 1)

    fig, (ax_keys, ax) = plt.subplots(
        1, 2, figsize=figsize, dpi=dpi,
        gridspec_kw={'width_ratios':[1, 12]}, sharey=True
    )

    labels_set = sorted(set([note[4] for note in score]))
    if isinstance(colors, list):
        colors = {label: colors[i % len(colors)] for i, label in enumerate(labels_set)}
    elif isinstance(colors, dict):
        pass  # already a dict
    else:
        # fallback: use a default colormap from matplotlib
        cmap = plt.get_cmap(str(colors))
        colors = {label: cmap(i / max(len(labels_set)-1, 1)) for i, label in enumerate(labels_set)}

    pitch_min = min(note[2] for note in score)
    pitch_max = max(note[2] for note in score)
    time_min = min(note[0] for note in score)
    time_max = max(note[0] + note[1] for note in score)

    for start, duration, pitch, velocity, label in score:
        if velocity_alpha is False:
            velocity = None
        rect = patches.Rectangle((start, pitch - 0.5), duration, 1, linewidth=1,
                                 edgecolor='k', facecolor=colors[label], alpha=velocity)
        ax.add_patch(rect)

    yticks = np.arange(pitch_min, pitch_max + 1)
    yticks_labels = [pretty_midi.utilities.note_number_to_name(p) for p in yticks]
    ax.set_ylim([pitch_min - 1.5, pitch_max + 1.5])
    ax.set_xlim([min(time_min, 0), time_max + 0.5])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend([patches.Patch(linewidth=1, edgecolor='k', facecolor=colors[key]) for key in labels_set],
              labels_set, loc='upper right', framealpha=1)
    
    draw_keyboard(ax_keys, pitch_min, pitch_max)

    if fig is not None:
        plt.tight_layout()

    if save:
        plt.savefig(f"{output_path}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

def generate_piano_roll(midi: pretty_midi.PrettyMIDI, output_path: str, \
                        figsize: tuple=(16,8), dpi: int=300, save: bool=True):
    """ 
        Generate piano roll

        Args:
            midi (pretty_midi.PrettyMIDI): prettyMIDI object
            output_path (str): Path to store the piano roll
    """
    csv = midi_to_csv(midi)
    score = csv_to_list(csv)
    visualize_piano_roll(score, output_path, figsize=figsize, dpi=dpi, save=save)


# ====================================
# Section 3: Conversion Utilities
# ====================================

def musescore_convert(input_path: str, output_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = str(Path(input_path).suffix)[1:] # we start from 1 to remove the dot
        shutil.copy(input_path, f"{tmpdir}/temp.{suffix}")

        # Update Environment Variables
        evs = os.environ.copy()
        evs["DISPLAY"] = ":0"
        evs["QT_QPA_PLATFORM"] = "offscreen"
        evs["XDG_RUNTIME_DIR"] = tmpdir
        evs["LD_LIBRARY_PATH"] = LD_PATH + evs.get("LD_LIBRARY_PATH", "")

        # perform conversion in a new environment
        subprocess.run(
            [MUSESCORE_PATH, "-o",  output_path, f"{tmpdir}/temp.{suffix}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=evs
        )

# Credits:`https://github.com/CPJKU/partitura`
def concatenate_images(
    filenames: Iterable[PathLike],
    out: Optional[PathLike] = None,
    concat_mode: str = "vertical",
) -> Optional[ImageFile]:
    """
    Concatenate Images to form one single image.

    Parameters
    ----------
    filenames: Iterable[PathLike]
        A list of images to be concatenated. This method assumes
        that all of the images have the same resolution and color mode
        (that is the case for png files generated by MuseScore).
        See `partitura.io.musescore.render_musescore`.
    out : Optional[PathLike]
        The output file where the image will be saved.
    concat_mode : {"vertical", "horizontal"}
        Whether to concatenate the images vertically or horizontally.
        Default is vertical.

    Returns
    -------
    new_image : Optional[PIL.Image.Image]
        The output image. This is only returned if `out` is not None.
    """
    # Check that concat mode is vertical or horizontal
    if concat_mode not in ("vertical", "horizontal"):
        raise ValueError(
            f"`concat_mode` should be 'vertical' or 'horizontal' but is {concat_mode}"
        )

    # Load images
    images = [Image.open(fn) for fn in filenames]

    # Get image sizes
    image_sizes = np.array([img.size for img in images], dtype=int)

    # size of the output image according to the concatenation mode
    if concat_mode == "vertical":
        output_size = (image_sizes[:, 0].max(), image_sizes[:, 1].sum())
    elif concat_mode == "horizontal":
        output_size = (image_sizes[:, 0].sum(), image_sizes[:, 1].max())

    # Color mode (assume it is the same for all images)
    mode = images[0].mode

    # DPI (assume that it is the same for all images)
    info = images[0].info

    # Initialize new image
    new_image = Image.new(mode=mode, size=output_size, color=0)

    # coordinates to place the image
    anchor_x = 0
    anchor_y = 0
    for img, size in zip(images, image_sizes):
        new_image.paste(img, (anchor_x, anchor_y))

        # update coordinates according to the concatenation mode
        if concat_mode == "vertical":
            anchor_y += size[1]

        elif concat_mode == "horizontal":
            anchor_x += size[0]

    # save image file
    if out is not None:
        new_image.save(out, **info)

    else:
        return new_image

def musescore_convert_img(input_path: str, output_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = str(Path(input_path).suffix)[1:] # we start from 1 to remove the dot
        shutil.copy(input_path, f"{tmpdir}/temp.{suffix}")

        # Update Environment Variables
        evs = os.environ.copy()
        evs["DISPLAY"] = ":0"
        evs["QT_QPA_PLATFORM"] = "offscreen"
        evs["XDG_RUNTIME_DIR"] = tmpdir
        evs["LD_LIBRARY_PATH"] = LD_PATH + evs.get("LD_LIBRARY_PATH", "")

        # perform conversion in a new environment
        # -T 10 trims the image by a margin of 10
        # -f is not ideal, but I do it because nakamura outputs can be weird after merging
        subprocess.run(
            [MUSESCORE_PATH, "-f", "-T", "10", "-o",  f"{tmpdir}/temp.png", f"{tmpdir}/temp.{suffix}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=evs
        )

        img_files = glob.glob(f"{tmpdir}/temp-*.png")
        concatenate_images(filenames=img_files, out=output_path, concat_mode="vertical")

# ============================================
# Section 4: MIDI-to-Score Inference Utilities
# ============================================

def nakamura_inference(perf_midi_path: str, score_midi_path: str, data_type: str="pop") -> None:
    """ 
        Perform inference with Nakamura score model.
        To avoid weird behaviour, pass absolute paths.

        Args:
            perf_midi_path (str): Path to performance midi file
            score_midi_path (str): Path to where to store the score (pass an absolute path)
            data_type (str): Pop/classical
    """
    # We do this to get rid of the ".mid" extension
    parent_perf, parent_stem = Path(perf_midi_path).parent, Path(perf_midi_path).stem
    parent_score, parent_score_stem = Path(score_midi_path).parent, Path(score_midi_path).stem

    perf_midi_path = parent_perf / parent_stem
    score_midi_path = parent_score / parent_score_stem
    cmd = ["bash", "./PerformanceMIDIToQuantizedMIDI_NL.sh", f"{perf_midi_path}", \
           f"{score_midi_path}", f"{data_type}"]
    subprocess.run(cmd, cwd="./externals/Nakamura/RQ")

def pm2s_inference(perfm_midi_path: str, score_midi_path: str):
    pm2s_processor = CRNNJointPM2S(
    beat_pps_args = {
        'prob_thresh': 0.5,
        'penalty': 1.0,
        'merge_downbeats': False,
        'method': 'dp',
    },
    ticks_per_beat = 480,
    notes_per_beat = [1, 6, 8],
    )

    # get the end time of the MIDI file
    temp = pretty_midi.PrettyMIDI(perfm_midi_path)
    end_time = temp.get_end_time()

    # Convert and save the generated score midi
    pm2s_processor.convert(perfm_midi_path, score_midi_path, start_time=0, end_time=end_time)

def beyer_midi_xml(midi_file: str, output_path: str) -> None:
    print("Starting inference...")
    model = Roformer.load_from_checkpoint("./extras/MIDI2ScoreTF.ckpt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x = MultistreamTokenizer.tokenize_midi(midi_file)
    length = x["pitch"].shape[0]
    y_hat = infer(x, model)
    mxl = MultistreamTokenizer.detokenize_mxl(y_hat)
    mxl = postprocess_score(mxl)
    # save the music21 score as a score
    mxl.metadata = m21.metadata.Metadata(title='')
    mxl.metadata.composer=''
    mxl.write('musicxml', fp=f'{output_path}', makeNotation=True)

# ============================================
# Section 5: Audio-to-MIDI Inference Utilities
# ============================================

def trans(audioPath: str, confPath: str="./extras/trans_config.conf", weight: str="./externals/Transkun/transkun/pretrained/2.0.pt", \
          segmentHopSize: Optional[float]=None, segmentSize: Optional[float]=None):
    """ 
        Transcribes an audio file using the transkun model

        Args:
        -----
            audioPath (str): path to audio file
            confPath (str): path to the model config
            weight (str): path to pretrained weights
            segmentHopSize (float): Not required. Default is in config
            segmentSize (float): Not required. Default is in config
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    confManager = moduleconf.parseFromFile(confPath)
    
    # For this to work, change the 
    TransKun = confManager["Model"].module.TransKun
    conf = confManager["Model"].config

    checkpoint = torch.load(weight, map_location = device)

    model = TransKun(conf = conf).to(device)
    # print("#Param(M):", computeParamSize(model))

    if not "best_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)

    model.eval()
    torch.set_grad_enabled(False)

    fs, audio= readAudio(audioPath)

    if(fs != model.fs):
        import soxr
        audio = soxr.resample(
                audio,          # 1D(mono) or 2D(frames, channels) array input
                fs,      # input samplerate
                model.fs# target samplerate
    )

    x = torch.from_numpy(audio).to(device)

    notesEst = model.transcribe(x, stepInSecond=segmentHopSize, segmentSizeInSecond=segmentSize, discardSecondHalf=False)

    outputMidi = writeMidi(notesEst)
    return outputMidi

# =========================================
# Section 6: Audio-to-MIDI Metric Utilities
# =========================================

def get_note_scores(pred_midi, gt_midi):
    """ 
        Get note-level scores for transcription
        and ground truth P-MIDI. Note that this
        function does not extend the note-offsets
        in order to simulate a piano performance.

        Args:
            pred_midi: (str|pretty_midi.PrettyMIDI) Transcribed MIDI
            gt_midi: (str|pretty_midi.PrettyMIDI) Predicted MIIDI
        
        Returns:
            {
                p: note-onset precision
                r: note-onset recall
                f: note-onset f1
                p_off: note-onset-offset precision
                r_off: note-onset-offset recall
                f_off: note-onset-offset f1
            }
    """
    # load ground truth midi
    if isinstance(gt_midi, str):
        gt_midi = pretty_midi.PrettyMIDI(gt_midi)
    else:
        gt_midi = gt_midi

    if isinstance(pred_midi, str):
        pred_midi = pretty_midi.PrettyMIDI(pred_midi)
    else:
        pred_midi = pred_midi

    # get notes for ground truth midi
    gt_midi_notes = []
    for instrument in gt_midi.instruments:
        for note in instrument.notes:
            gt_midi_notes.append((note.start, note.end, note.pitch, note.velocity))
    gt_midi_arr = np.array(gt_midi_notes)

    # get notes for predicted midi
    pred_midi_notes = []
    for instrument in pred_midi.instruments:
        for note in instrument.notes:
            pred_midi_notes.append((note.start, note.end, note.pitch, note.velocity))
    pred_midi_arr = np.array(pred_midi_notes)
    pred_ints = pred_midi_arr[:, :2]
    pred_pitches = pred_midi_arr[:, 2]
    gt_ints = gt_midi_arr[:, :2]
    gt_pitches = gt_midi_arr[:, 2]
    p, r, f, _ = prf(gt_ints, gt_pitches, pred_ints, pred_pitches, offset_ratio=None)
    p_off, r_off, f_off, _ = prf(gt_ints, gt_pitches, pred_ints, pred_pitches)
    result = {
        'p': p,
        'r': r,
        'f': f,
        'p_off': p_off,
        'r_off': r_off,
        'f_off': f_off
    }
    return result

def peamt_score(pred_midi: str, gt_midi: str) -> float:
    """ 
        Calculates the PEAMT score. PEAMT is a perceptual-based
        metric that tells us how similar a score is with the 
        ground-truth score.

        Args:
            pred_midi (str): Transcribed MIDI file
            gt_midi (str): Ground truth MIDI file

        Return:
            value (float): Measure of similarity (between 0 and 1)
    """
    eval = PEAMT()
    value = eval.evaluate_from_midi(gt_midi, pred_midi)
    return value

def compute_mpteval(pred: str, target: str, result_type: str="harmony") -> dict | np.ndarray:
    """ 
        Compute mpteval metrics.

        Args:
            pred (str): Predicted MIDI
            target (str): Ground truth MIDI
            result_type (str): Default is harmony. Other option is all.

        Returns:
            if result_type is harmony, returns an arr of 
            (cloud_diameter, cloud_momentum, tensile_strain).

            If result_type is all, returns a dict of all
            metrics.
    """

    result_dict = {'melody_ioi_corr': 0, 'acc_ioi_corr': 0, \
                   'ratio_ioi_corr': 0, 'cloud_diameter_corr': 0, \
                    'cloud_momentum_corr': 0, 'tensile_strain_corr': 0}
    
    ref_perf = pt.load_performance_midi(target)
    pred_perf = pt.load_performance_midi(pred)

    timing_metrics = mpteval.timing.timing_metrics_from_perf(ref_perf, pred_perf)

    # we use a window size of 1 for the harmony metrics because we are dealing with
    # small segments
    harmony_metrics = mpteval.harmony.harmony_metrics_from_perf(ref_perf, pred_perf, ws=1)

    timing_metrics_dict = {}
    harmony_metrics_dict = {}
    for name in timing_metrics.dtype.names:
        timing_metrics_dict[name] = timing_metrics[name][0]

    for name in harmony_metrics.dtype.names:
        harmony_metrics_dict[name] = harmony_metrics[name][0]

    for key in result_dict.keys():
        if key in timing_metrics_dict:
            result_dict[key] = timing_metrics_dict[key]
        elif key in harmony_metrics_dict:
            result_dict[key] = harmony_metrics_dict[key]
    
    # check if corr_diameter is nan
    for key in result_dict.keys():
        if np.isnan(result_dict[key]):
            # set to 0 and print warning
            print(f"Warning: {key} is nan, skipping guilty MIDI file: {target}, pred: {pred}")
            raise RuntimeError('Nan value obtained!')
            #result_dict[key] = 0.0
        
    if result_type == "harmony":
        return harmony_metrics
    else:
        return result_dict
    
# Credits: https://github.com/Yujia-Yan/Transkun/blob/main/transkun/Evaluation.py
def intersectTwoInterval(intervalA, intervalB):
    l = max(intervalA[0], intervalB[0])
    r = min(intervalA[1], intervalB[1])
    return (l,r)

def findIntersectListOfIntervals(listA, listB):
    i = 0
    j = 0
    result = []
    while i<len(listA) and j<len(listB):
        l,r = intersectTwoInterval(listA[i], listB[j])
        if r>=l:
            # check if (l,r) can be merged into the previous one
            if len(result)>0 and result[-1][1] == l:
                result[-1] = (result[-1][0],r)
            else:
                result.append((l,r))
        
        if listA[i][1] < listB[j][1]:
            i = i+1
        else:
            j = j+1

        
    
    return result
    
def computeIntervalLengthSum(intervals, countZero=True):
    s = 0
    if countZero:
        prevEnd = -1
        for e in intervals:
            s+= e[1]-e[0]
            if prevEnd < e[0]:
                s+= 1

            prevEnd = e[1]
    else:
        for e in intervals:
            s+= e[1]-e[0]

    return s

def compareFramewise(intervalEst, intervalGT, countZero=True):
    nEst = computeIntervalLengthSum(intervalEst, countZero)
    nGT = computeIntervalLengthSum(intervalGT, countZero)
    intersected = findIntersectListOfIntervals(intervalEst,intervalGT)
    nIntersected = computeIntervalLengthSum(intersected, countZero)
    nUnion = nGT+nEst- nIntersected

    return nGT,nEst, nIntersected

def compute_activation_metrics(pred: str|pretty_midi.PrettyMIDI, gt: str|pretty_midi.PrettyMIDI):
    """ 
        Computes the activation-level metrics. 
        Note that this does not extend pedals as is
        based on models trained without pedal extension.

        Args:
            pred (str | pretty_midi.PrettyMIDI): Path to MIDI transcription or pretty MIDI object
            gt (str | pretty_midi.PrettyMIDI): Path to ground-truth MIDI file or pretty MIDI object
        
        Returns:
            out (tuple): (precision, recall, f1-score)
    """
    if isinstance(gt, pretty_midi.PrettyMIDI):
        gt = gt 
    else:
        gt = pretty_midi.PrettyMIDI(gt)

    if isinstance(pred, pretty_midi.PrettyMIDI):
        pred = pred 
    else:
        pred = pretty_midi.PrettyMIDI(pred)

    # get notes for ground truth midi
    gt_midi_notes = defaultdict(list)
    for instrument in gt.instruments:
        for note in instrument.notes:
            gt_midi_notes[note.pitch].append(note)

    # get notes for predicted midi
    pred_midi_notes = defaultdict(list)
    for instrument in pred.instruments:
        for note in instrument.notes:
            pred_midi_notes[note.pitch].append(note)

    pred_ints = []
    for i in range(0, 128):
        ints = []
        for n in pred_midi_notes[i]:
            ints.append((n.start, n.end))
        pred_ints.append(ints)

    gt_ints = []
    for i in range(0, 128):
        ints = []
        for n in gt_midi_notes[i]:
            ints.append((n.start, n.end))
        gt_ints.append(ints)
    
    num_gt = 0
    num_pred = 0
    num_correct = 0
    for ints_a, ints_b in zip(pred_ints, gt_ints):
        curr_num_gt, curr_num_pred, curr_num_corr = compareFramewise(ints_a, ints_b, countZero=False)
        num_gt += curr_num_gt
        num_pred += curr_num_pred
        num_correct += curr_num_corr
    
    p = num_correct/(num_pred + 1e-8)
    r = num_correct/(num_gt + 1e-8)
    f = (2*num_correct)/(num_pred + num_gt + 1e-8)
    return p, r, f


# =========================================
# Section 7: MIDI-to-Score Metric Utilities
# =========================================

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

def get_seq_hands(tokens: torch.Tensor) -> list:
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

    # Create sequences for left and right hand
    seq = []
    left_hand_seq = []
    right_hand_seq = []
    for i in range(tokens['offset'].shape[0]):
        parts = []
        for key in tokens.keys():
            print(key, tokens[key], tokens[key].shape)
            exit(1)
            if key not in ['offset', 'duration', 'pitch', 'hand']:
                continue
            prefix = TOKEN_NAME_MAP[key]
            parts.append(f"{prefix}{int(stream_indices[key][i])}")
        token = '_'.join(parts)
        seq.append(token)
    return left_hand_seq, right_hand_seq

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
    gt_left, gt_right = MusicXML_to_tokens(gt_xml)
    pred_left, pred_right = MusicXML_to_tokens(pred_xml)

    wer_left = compute_ed_metrics([gt_left], [pred_left])["sym-er"]
    wer_right = compute_ed_metrics([gt_right], [pred_right])["sym-er"]
    wer = round((wer_left + wer_right)/2, 4)
    return {'sym-er': wer}


# =========================================
# Section 8: Dataset Utilities
# =========================================
def get_xml_score_revamp(score: m21.stream.Score, measure_nums: list, key_sig: list, time_sig: list) -> m21.stream.Score:
    """
        Gets measures from a ground truth score and
        returns this as a new score.

        Args:
        ------
            score (m21.stream.Score): Ground truth score 
            measure_nums (list): Measures to extract
            key_sig (list): List of key signature changes
            time_sig (list): List of time signature changes

        Returns:
        --------
            new_score (m21.stream.Score): Returns the new score
    """
    # We might run into problems because of pickups
    first_measure = score.parts[0].getElementsByClass(m21.stream.Measure).first()
    has_pickup = (first_measure is not None) and (first_measure.number == 0)
    if has_pickup:
        print("Warning! Piece has pickup. Adjusting Measure Numbers....")
        measure_nums = [measure_nums[0]-1, measure_nums[-1]-1]

    # load the xml file with music21
    parts = list(score.parts)
    new_score = m21.stream.Score()

    for p in parts:
        # create a new part 
        new_part = m21.stream.Part()
        new_part.id = p.id
        
        for idx, measure_num in enumerate(range(measure_nums[0], measure_nums[-1] + 1)):
            m = p.measure(measure_num)

            # if a tempo marking exists in the measure we will capture it, 
            # But that is not often the case
            tempo_markings = m.getContextByClass(m21.tempo.MetronomeMark)
            if tempo_markings:
                if tempo_markings not in m.recurse().getElementsByClass(m21.tempo.MetronomeMark):
                    m.insert(0, tempo_markings)

            if idx == 0:
                clef = m.getContextByClass(m21.clef.Clef)
                # insert clef at the beginning of the measure
                if clef:
                    if clef not in m.recurse().getElementsByClass(m21.clef.Clef):
                        m.insert(0, clef)
                else:
                    print(f"Warning: No clef found in measure {measure_num} of part {p.id}")


            # create time signature and key signature objects at the beginning of the measure
            if idx == 0:
                ts = m21.meter.TimeSignature(time_sig[idx][0])
                ks = m21.key.KeySignature(key_sig[idx][1])
                m.insert(0, ts)
                m.insert(0, ks)
            else:
                # Create time and signature objects only if they change
                if time_sig[idx] != time_sig[idx-1]:
                    ts = m21.meter.TimeSignature(time_sig[idx][0])
                    m.insert(0, ts)
                if key_sig[idx] != key_sig[idx-1]:
                    ks = m21.key.KeySignature(key_sig[idx][1])
                    m.insert(0, ks)

            new_part.append(m)

        new_score.append(new_part)

    return new_score

def get_xml_score(score: m21.stream.Score, measure_nums: list, key_sig: list, time_sig: list) -> m21.stream.Score:
    """
        Gets measures from a ground truth score and
        returns this as a new score.

        Args:
        ------
            score (m21.stream.Score): Ground truth score 
            measure_nums (list): Measures to extract
            key_sig (list): List of key signature changes
            time_sig (list): List of time signature changes

        Returns:
        --------
            new_score (m21.stream.Score): Returns the new score
    """
    # We might run into problems because of pickups
    first_measure = score.parts[0].getElementsByClass(m21.stream.Measure).first()
    has_pickup = (first_measure is not None) and (first_measure.number == 0)
    if has_pickup:
        print("Warning! Piece has pickup. Adjusting Measure Numbers....")
        measure_nums = [measure_nums[0]-1, measure_nums[-1]-1]

    # load the xml file with music21
    parts = list(score.parts)
    new_score = m21.stream.Score()

    for p in parts:
        # create a new part 
        new_part = m21.stream.Part()
        new_part.id = p.id
        
        for idx, measure_num in enumerate(range(measure_nums[0], measure_nums[-1] + 1)):
            m = p.measure(measure_num)
            new_m = m21.stream.Measure(number=idx+1) # new measure

            # if a tempo marking exists in the measure we will capture it, 
            # But that is not often the case
            tempo_markings = m.getContextByClass(m21.tempo.MetronomeMark)
            if tempo_markings:
                if tempo_markings not in m.recurse().getElementsByClass(m21.tempo.MetronomeMark):
                    #m.insert(0, tempo_markings)
                    new_m.insert(0, tempo_markings)

            if idx == 0:
                clef = m.getContextByClass(m21.clef.Clef)
                # insert clef at the beginning of the measure
                if clef:
                    if clef not in m.recurse().getElementsByClass(m21.clef.Clef):
                        #m.insert(0, clef)
                        new_m.insert(0, clef)
                else:
                    print(f"Warning: No clef found in measure {measure_num} of part {p.id}")


            # create time signature and key signature objects at the beginning of the measure
            if idx == 0:
                ts = m21.meter.TimeSignature(time_sig[idx][0])
                ks = m21.key.KeySignature(key_sig[idx][1])
                # m.insert(0, ts)
                # m.insert(0, ks)
                new_m.insert(0, ts)
                new_m.insert(0, ks)
            else:
                # Create time and signature objects only if they change
                if time_sig[idx] != time_sig[idx-1]:
                    ts = m21.meter.TimeSignature(time_sig[idx][0])
                    #m.insert(0, ts)
                    new_m.insert(0, ts)
                if key_sig[idx] != key_sig[idx-1]:
                    ks = m21.key.KeySignature(key_sig[idx][1])
                    #m.insert(0, ks)
                    new_m.insert(0, ks)
            
            for el in m:
                if isinstance(el, m21.stream.Voice):
                    voice = m21.stream.Voice()
                    for item in el:
                        voice.insert(item.offset, item)
                    assert voice.isWellFormedNotation()
                    new_m.insert(el.offset, voice)
                else:
                    new_m.insert(el.offset, el)
            
            # voices = set()
            # for el in m:
            #     if isinstance(el, m21.stream.Voice):
            #         voices.add(el._id)
            
            # # This seems to fix MUSTER bug. Test extensively later on...
            # voice_map = {old: new for new, old in enumerate(sorted(list(voices)), start=1)}
            # for el in m:
            #     if isinstance(el, m21.stream.Voice):
            #         el._id = voice_map[el._id]

            #new_part.append(m)
            new_part.append(new_m)

        new_score.append(new_part)

    return new_score


def get_midi_note_events_strict(midi: pretty_midi.PrettyMIDI, start: float, end: float) -> np.ndarray:
    """
        Gets MIDI note events within a window in a
        strict manner.

        Args:
        ------
            midi (pretty_midi.PrettyMIDI): prettyMIDI object
            start (float): start time of window
            end (float): end time of window

        Returns:
        --------
            res (np.ndarray): Array of note events
    """
    note_events = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                if note.start >= end or note.end <= start:
                    continue 

                # neglect note durations which are about 50ms close to the
                # end of the segment (it is unlikely to be captured in the audio properly) 
                if end - note.start < 0.05:
                    continue

                pitch = note.pitch

                # Need to deal with the case where the note starts before
                # the start of the segment and ends after the end of the segment
                # Not sure if what I have done below is the right thing to do
                note_events.append([note.start - start, min(note.end - start, end - start), pitch, note.velocity])

    # Get the tempo changes    
    return np.array(note_events)

def get_tsig_changes(midi: pretty_midi.PrettyMIDI, start: float, \
        end: float) -> list:
    """
        Get time signature changes for a window segment

        Args:
        -----
            midi (pretty_midi.PrettyMIDI): pretty_midi object
            start (float): Start time
            end (float): End time

        Returns:
        -------
            new_time_sigs (list): List of new time signature changes
    """
    time_sigs = midi.time_signature_changes
    new_time_sigs = []

    # Find time signature at start time
    ts_before_start = [ts for ts in time_sigs if ts.time <= start]
    if len(ts_before_start) > 0:
        ts_start = ts_before_start[-1]
    else:
        # Fallback
        ts_start = pretty_midi.TimeSignature(4, 4, 0)

    # Add ts_start
    new_time_sigs.append(
        pretty_midi.TimeSignature(ts_start.numerator, ts_start.denominator, 0)
    )

    # Add all tsigs in the window of interest
    for ts in time_sigs:
        if start < ts.time < end:
            new_time_sigs.append(
                pretty_midi.TimeSignature(ts.numerator, ts.denominator, ts.time - start)
            )
    return new_time_sigs

def get_ksig_changes(midi: pretty_midi.PrettyMIDI, start: float, \
        end: float) -> list:
    """
        Get key signature changes for a window segment

        Args:
        -----
            midi (pretty_midi.PrettyMIDI): pretty_midi object
            start (float): Start time
            end (float): End time

        Returns:
        -------
            new_key_sigs (list): List of new time signature changes
    """
    key_sigs = midi.key_signature_changes
    new_key_sigs = []

    # Find key signature at start time
    ks_before_start = [ks for ks in key_sigs if ks.time <= start]
    if len(ks_before_start) > 0:
        ks_start = ks_before_start[-1]
    else:
        # Fallback
        ks_start = pretty_midi.KeySignature(key_number=0, time=start)

    # Add ks_start
    new_key_sigs.append(
        pretty_midi.KeySignature(key_number=ks_start.key_number, time=0)
    )

    # Add all ksigs in the window of interest
    for ks in key_sigs:
        if start < ks.time < end:
            new_key_sigs.append(
                pretty_midi.KeySignature(key_number=ks.key_number, time=ks.time - start)
            )
    return new_key_sigs

def get_tempo_changes(midi: pretty_midi.PrettyMIDI, start: float, \
        end: float) -> tuple:
    """
        Get tempo changes for a window segment

        Args:
        -----
            midi (pretty_midi.PrettyMIDI): PrettyMIDI object
            start (float): start time
            end (float): end time

        Returns:
        --------
            res (tuple[np.ndarray, np.ndarray]): new_tempo_times, new_tempi
    """
    tempo_times, tempi = midi.get_tempo_changes()
    new_tempo_times = []
    new_tempi = []

    # Find tempo at start time
    idxs_before_start = np.where(tempo_times <= start)[0]
    if len(idxs_before_start) > 0:
        last_idx = idxs_before_start[-1]
        tempo_at_start = tempi[last_idx]
    else:
        # Fallback
        tempo_at_start = tempi[0] if len(tempi) > 0 else 120.0

    new_tempo_times.append(0.0)
    new_tempi.append(tempo_at_start)

    # All tempo changes in the window
    valid_idxs = np.where((tempo_times > start) & (tempo_times < end))[0]
    for idx in valid_idxs:
        new_time = tempo_times[idx] - start
        new_tempo_times.append(new_time)
        new_tempi.append(tempi[idx])

    return np.array(new_tempo_times), np.array(new_tempi)

def update_tempo_changes(midi_obj, tempo_time, tempi):
    last_tick = 0
    last_tick_scale = 60.0/(tempi[0].item() * midi_obj.resolution)
    previous_time = 0.
    midi_obj._tick_scales = [(last_tick, last_tick_scale)]

    for i in range(1, len(tempo_time)):
        # compute new tick position
        tick = last_tick + (tempo_time[i].item() - previous_time)/last_tick_scale 
        # Update tick scale
        tick_scale = 60.0 / (tempi[i].item() * midi_obj.resolution)
        # Don't add repeat tick scales
        if tick_scale != last_tick_scale:
            midi_obj._tick_scales.append((int(round(tick)), tick_scale))
            previous_time = tempo_time[i].item()
            last_tick, last_tick_scale = tick, tick_scale

def merge_midi(midi_in: str, midi_out: str):
    midi_obj = pretty_midi.PrettyMIDI(midi_in)
    midi_merged = pretty_midi.PrettyMIDI()
    midi_merged.resolution = midi_obj.resolution
    midi_merged.time_signature_changes = midi_obj.time_signature_changes[:]
    midi_merged.key_signature_changes = midi_obj.key_signature_changes[:]
    tempo_changes = get_tempo_changes(midi_obj, 0, midi_obj.get_end_time())
    update_tempo_changes(midi_merged, tempo_changes[0], tempo_changes[1])

    inst = pretty_midi.Instrument(program=0) # piano

    for track in midi_obj.instruments:
        for n in track.notes:
            #inst.notes.append(n)
            inst.notes.append(
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start,
                    end=n.end
                )
            )

    
    midi_merged.instruments.append(inst)
    midi_merged.write(midi_out)
    return midi_merged

# ==============================
# Section 9: Tokenization utils.
# ==============================

# Credits: `https://github.com/TimFelixBeyer/ScoreTransformer/blob/main/score_transformer/tokenizer/score_to_tokens.py`
def attributes_to_tokens(attributes, staff=None):
    """tokenize 'attributes' section in MusicXML"""
    tokens = []
    divisions = None

    for child in attributes.contents:
        match child.name:
            case 'divisions':
                divisions = int(child.text)
            case 'clef' | 'key' | 'time':
                if staff is not None:
                    if 'number' in child.attrs and int(child['number']) != staff:
                        break
                tokens.append(attribute_to_token(child))

    return tokens, divisions

def attribute_to_token(child):
    """clef, key signature, and time signature"""
    match child.name:
        case 'clef':
            if child.sign.text == 'G':
                return 'clef_treble'
            elif child.sign.text == 'F':
                return 'clef_bass'
        case 'key':
            key = int(child.fifths.text)
            if key < 0:
                return f'key_flat_{abs(key)}'
            elif key > 0:
                return f'key_sharp_{key}'
            else:
                return f'key_natural_{key}'
        case 'time':
            times = [int(c.text) for c in child.contents if isinstance(c, Tag)] # excluding '\n'
            if times[1] == 2:
                return f'time_{times[0]*2}/{times[1]*2}'
            elif times[1] > 4:
                fraction = str(Fraction(times[0], times[1]))
                if int(fraction.split('/')[1]) == 2: # X/2
                    return f"time_{int(fraction.split('/')[0])*2}/{int(fraction.split('/')[0])*2}"
                else:
                    return 'time_' + fraction
            else:
                return f'time_{times[0]}/{times[1]}'

def aggregate_notes(voice_notes):
    """notes to chord"""
    for note in voice_notes[1:]:
        if note.chord is not None:
            last_note = note.find_previous('note')
            last_note.insert(0, note.pitch)
            note.decompose()

def note_to_tokens(note, divisions, note_name=True):
    """notes and rests"""
    beam_translations = {'begin': 'start', 'end': 'stop', 'forward hook': 'partial-right', 'backward hook': 'partial-left'}

    if note.duration is None: # gracenote
        return []

    duration_in_fraction = str(Fraction(int(note.duration.text), divisions))

    if note.rest:
        return ['rest', f'len_{duration_in_fraction}'] # for rests

    tokens = []

    # pitches
    for pitch in note.find_all('pitch'):
        if note_name:
            if pitch.alter:
                alter_to_symbol= {'-2': 'bb', '-1': 'b', '0':'', '1': '#', '2': '##'}
                tokens.append(f"note_{pitch.step.text}{alter_to_symbol[pitch.alter.text]}{pitch.octave.text}")
            else:
                tokens.append(f"note_{pitch.step.text}{pitch.octave.text}")
        else:
            note_number = pretty_midi.note_name_to_number(pitch.step.text + pitch.octave.text) # 'C4' -> 60
            if pitch.alter:
                note_number += int(pitch.alter.text)
            tokens.append(f'note_{note_number}')

    # len
    tokens.append(f'len_{duration_in_fraction}')

    if note.stem:
        tokens.append(f'stem_{note.stem.text}')

    if note.beam:
        beams = note.find_all('beam')
        tokens.append('beam_' + '_'.join([beam_translations[b.text] if b.text in beam_translations else b.text for b in beams]))

    if note.tied:
        tokens.append('tie_' + note.tied.attrs['type'])

    return tokens

def element_segmentation(measure, soup, staff=None):
    """divide elements into three sections"""
    voice_starts, voice_ends = {}, {}
    position = 0
    for element in measure.contents:
        if element.name == 'note':
            if element.duration is None: # gracenote
                continue
            if element.voice is not None:
                voice = element.voice.text
            else:
                voice = 0
            duration = int(element.duration.text)
            if element.chord: # rewind for concurrent notes
                position -= last_duration

            if element.staff and int(element.staff.text) == staff:
                voice_starts[voice] = min(voice_starts[voice], position) if voice in voice_starts else position
                start_tag = soup.new_tag('start')
                start_tag.string = str(position)
                element.append(start_tag)

            position += duration

            if element.staff and int(element.staff.text) == staff:
                voice_ends[voice] = max(voice_ends[voice], position) if voice in voice_ends else position
                end_tag = soup.new_tag('end')
                end_tag.string = str(position)
                element.append(end_tag)

            last_duration = duration
        elif element.name == 'backup':
            position -= int(element.duration.text)
        elif element.name == 'forward':
            position += int(element.duration.text)
        else: # other types
            start_tag = soup.new_tag('start')
            end_tag = soup.new_tag('end')

            start_tag.string = str(position)
            end_tag.string = str(position)

            element.append(start_tag)
            element.append(end_tag)

    # voice section
    voice_start = sorted(voice_starts.values())[1] if voice_starts else 0
    voice_end = sorted(voice_ends.values(), reverse=True)[1] if voice_ends else 0

    pre_voice_elements, post_voice_elements, voice_elements = [], [], []
    for element in measure.contents:
        if element.name in ('backup', 'forward'):
            continue
        if element.name == 'note' and element.duration is None: # gracenote
            continue
        if staff is not None:
            if element.staff and int(element.staff.text) != staff:
                continue

        if voice_starts or voice_ends:
            if int(element.end.text) <= voice_start:
                pre_voice_elements.append(element)
            elif voice_end <= int(element.start.text):
                post_voice_elements.append(element)
            else:
                voice_elements.append(element)
        else:
            pre_voice_elements.append(element)

    return pre_voice_elements, voice_elements, post_voice_elements

def measures_to_tokens(measures, soup, staff=None, note_name=True):
    divisions = 0
    tokens = []
    for measure in measures:

        tokens.append('bar')
        if staff is not None:
            notes = [n for n in measure.find_all('note') if n.staff and int(n.staff.text) == staff]
        else:
            notes = measure.find_all('note')

        voices = list(set([n.voice.text for n in notes if n.voice]))
        for voice in voices:
            voice_notes = [n for n in notes if n.voice and n.voice.text == voice]
            aggregate_notes(voice_notes)

        if len(voices) > 1:
            pre_voice_elements, voice_elements, post_voice_elements = element_segmentation(measure, soup, staff)

            for element in pre_voice_elements:
                if element.name == 'attributes':
                    attr_tokens, div = attributes_to_tokens(element, staff)
                    tokens += attr_tokens
                    divisions = div if div else divisions
                elif element.name == 'note':
                    tokens += note_to_tokens(element, divisions, note_name)

            if voice_elements:
                for voice in voices:
                    tokens.append('<voice>')
                    for element in voice_elements:
                        if (element.voice and element.voice.text == voice) or (not element.voice and voice == '1'):
                            if element.name == 'attributes':
                                attr_tokens, div = attributes_to_tokens(element, staff)
                                tokens += attr_tokens
                                divisions = div if div else divisions
                            elif element.name == 'note':
                                tokens += note_to_tokens(element, divisions, note_name)
                    tokens.append('</voice>')

            for element in post_voice_elements:
                if element.name == 'attributes':
                    attr_tokens, div = attributes_to_tokens(element, staff)
                    tokens += attr_tokens
                    divisions = div if div else divisions
                elif element.name == 'note':
                    tokens += note_to_tokens(element, divisions, note_name)
        else:
            for element in measure.contents:
                if staff is not None:
                    if element.name in ('attributes', 'note') and element.staff and int(element.staff.text) != staff:
                        continue
                if element.name == 'attributes':
                    attr_tokens, div = attributes_to_tokens(element, staff)
                    tokens += attr_tokens
                    divisions = div if div else divisions
                elif element.name == 'note':
                    tokens += note_to_tokens(element, divisions, note_name)

    return tokens

def MusicXML_to_tokens(mxl, note_name=True) -> list[str]:
    if type(mxl) is str:
        try:
            with open(mxl, "r") as f:
                soup = BeautifulSoup(f, 'lxml-xml', from_encoding='utf-8')
        except UnicodeDecodeError as e:
            print(mxl, e)
            folder = uuid.uuid4()
            try:
                with zipfile.ZipFile(mxl, 'r') as zip_ref:
                    zip_ref.extractall(f"/tmp/{folder}")
                file = [f for f in os.listdir(f"/tmp/{folder}") if f.endswith(".xml") or f.endswith(".musicxml")][0]
                with open(f"/tmp/{folder}/" + file, "r") as f:
                    soup = BeautifulSoup(f, 'lxml-xml', from_encoding='utf-8')
            finally:
                shutil.rmtree(f"/tmp/{folder}")

    for tag in soup(string='\n'): # eliminate line breaks
        tag.extract()

    parts = [part.find_all('measure') for part in soup.find_all('part')]

    if len(parts) == 1:
        tokens = ['R'] + measures_to_tokens(parts[0], soup, staff=1, note_name=note_name)
        tokens += ['L'] + measures_to_tokens(parts[0], soup, staff=2, note_name=note_name)
        R_tokens = ['R'] + measures_to_tokens(parts[0], soup, staff=1, note_name=note_name)
        L_tokens = ['L'] + measures_to_tokens(parts[0], soup, staff=2, note_name=note_name)
    elif len(parts) == 2:
        tokens = ['R'] + measures_to_tokens(parts[0], soup, note_name=note_name)
        tokens += ['L'] + measures_to_tokens(parts[1], soup, note_name=note_name)
        R_tokens = ['R'] + measures_to_tokens(parts[0], soup, note_name=note_name)
        L_tokens = ['L'] + measures_to_tokens(parts[1], soup, note_name=note_name)
    else:
        tokens = ['R'] + measures_to_tokens(parts[0], soup, note_name=note_name)
        tokens += ['L'] + measures_to_tokens(parts[1], soup, note_name=note_name)
        R_tokens = ['R'] + measures_to_tokens(parts[0], soup, note_name=note_name)
        L_tokens = ['L'] + measures_to_tokens(parts[1], soup, note_name=note_name)
        print(f'WARNING: Piano MusicXML must have 1 or 2 parts, not {len(parts)} - using first two parts only.')

    return R_tokens, L_tokens

# Parse MusicXML files to check for broken ties
#path = "./data/score_examples/pred_xml/BLINOV04M_m39_beyer.musicxml"
path = "./data/xml_score/BLINOV04M_m39.musicxml"

# Algorithm to detect broken ties...
# Uses a hashmap; memory complexity O(n), time complexity O(n)
# tie_dict has (key: noteName, value: LinkedList of tie connections)

class LinkedList:
    def __init__(self, head):
        self.head = head
        self.last = head
    
    def append(self, node):
        self.last.next = node
        self.last = node

class Node:
    def __init__(self, el, next=None):
        self.el = el
        self.next = next
    
    def __str__(self):
        return f"{self.el.nameWithOctave}"

    def __repr__(self):
        return self.__str__()

def remove_broken_ties(path: str|m21.stream.Stream, output_path: str):
    if isinstance(path, str):
        s = m21.converter.parse(path)
    else:
        s = path

    broken_ties = []
    tie_dict = defaultdict(LinkedList)
    for el in list(s.recurse()):
        if isinstance(el, m21.note.Note) or isinstance(el, m21.chord.Chord):
            notes = []
            if isinstance(el, m21.chord.Chord):
                for n in el.notes:
                    notes.append(n)
            else:
                notes.append(el)
            
            for n in notes:
                if n.tie is None:
                    continue
                tie_type = n.tie.type

                if tie_type == "start":
                    tie_dict[n.nameWithOctave] = LinkedList(Node(n))
                
                if tie_type == "continue":
                    ll = tie_dict.get(n.nameWithOctave, None)
                    if ll is None:
                        # This is a broken tie
                        broken_ties.append(n)
                    else:
                        ll.append(Node(n))
                
                if tie_type == "stop":
                    ll = tie_dict.get(n.nameWithOctave, None)
                    if ll is None:
                        # This is a broken tie
                        broken_ties.append(n)
                    else:
                        ll.append(Node(n))

    # Now we have to pass the dict to add elements that have a start tie but no end tie
    for key in tie_dict.keys():
        ll = tie_dict[key]
        last_node = ll.last
        curr_node = ll.head
        if last_node.el.tie is None:
            continue
        else:
            tie_type = last_node.el.tie.type
            if tie_type != "stop":
                # Add all of the notes from the head node
                # to broken ties list
                while curr_node:
                    broken_ties.append(curr_node.el)
                    curr_node = curr_node.next
                    if curr_node is None:
                        break

    for n in broken_ties:
        n.tie = None

    s.write('musicxml', fp = f"{output_path}", makeNotation=True)

def merge_score(xml_path: str, store_path: str):
    s = m21.converter.parse(xml_path)
    rh = m21.stream.Part()
    lh = m21.stream.Part()

    # Append treble clefs and bass clefs
    rh.insert(0, m21.clef.TrebleClef())
    lh.insert(0, m21.clef.BassClef())

    parts = list(s.parts)
    assert len(parts) == 4, "Only four (4) parts are acceptable."

    for i in range(2):
        # i = 0 will be for rh
        # i = 1 will be for lh
        if i == 0:
            voice1 = parts[i]
            voice2 = parts[i+1]
        else:
            voice1 = parts[i+1]
            voice2 = parts[i+2]
        
        # Now we need to merge these two voices into one voice
        mv1 = voice1.getElementsByClass('Measure')
        mv2 = voice2.getElementsByClass('Measure')
        assert len(mv1) == len(mv2), "Voice measures should be equal!"

        for j in range(len(mv1)-2): # skip the last two measures for now
            # Get the jth measure for each voice
            m1 = mv1[j]
            m2 = mv2[j]
            v1 = m21.stream.Voice()
            v2 = m21.stream.Voice()
            new_m = m21.stream.Measure(number=j+1)

            if len(m1.getElementsByClass([m21.note.Note, m21.chord.Chord])) > 0:
                for el in m1:
                    if isinstance(el, m21.clef.Clef):
                        continue
                    elif isinstance(el, m21.key.KeySignature) or isinstance(el, m21.meter.TimeSignature):
                        new_m.insert(el.offset, deepcopy(el))
                    else:
                        if isinstance(el, m21.note.Note):
                            el.stemDirection = 'unspecified'
                        
                        if isinstance(el, m21.tempo.MetronomeMark):
                            new = m21.tempo.MetronomeMark(numberSounding=el.number)
                            new.style.hideObjectOnPrint = True
                            new_m.insert(el.offset, new)
                            continue
                        
                        v1.insert(el.offset, deepcopy(el))
            
            if len(m2.getElementsByClass([m21.note.Note, m21.chord.Chord])) > 0:
                for el in m2:
                    if isinstance(el, m21.clef.Clef):
                        continue
                    elif isinstance(el, m21.key.KeySignature) or isinstance(el, m21.meter.TimeSignature):
                        new_m.insert(el.offset, deepcopy(el))
                    else:
                        if isinstance(el, m21.note.Note):
                            el.stemDirection = 'unspecified'

                        if isinstance(el, m21.tempo.MetronomeMark):
                            new = m21.tempo.MetronomeMark(numberSounding=el.number)
                            new.style.hideObjectOnPrint = True
                            new_m.insert(el.offset, new)
                            continue

                        v2.insert(el.offset, deepcopy(el))
            
            if len(v1) > 0:
                new_m.insert(0, v1)
            if len(v2) > 0:
                new_m.insert(0, v2)

            if i == 0:
                rh.append(new_m)
            else:
                lh.append(new_m)
    
    new_score = m21.stream.Score()
    new_score.append(rh)
    new_score.append(lh)
    
    assert new_score.isWellFormedNotation(), "Stream is not well-formed."
    new_score.write("musicxml", fp=f"{store_path}", makeNotation=True)

def strip_tempo_markings(xml_path: str, store_path: str):
    # For scores we postprocess to hide the tempo, the audio
    # should be generated from the MIDI scores for perceptual 
    # similarity to the audio
    s = m21.converter.parse(xml_path)

    # Our goal is to use this function to do some postprocessing
    # For now, we are looking at hiding tempo markings
    for el in list(s.recurse().getElementsByClass(m21.tempo.MetronomeMark)):
        # new_marking = m21.tempo.MetronomeMark(numberSounding=el.number)
        # new_marking.style.hideObjectOnPrint = True

        # Get site
        site = el.activeSite
        site.remove(el)
    
    s.metadata = m21.metadata.Metadata(title='')
    s.metadata.composer=''
    s.write("musicxml", fp=f"{store_path}")
