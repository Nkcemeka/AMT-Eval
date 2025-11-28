from typing import Optional
import moduleconf
import torch
import pretty_midi
import numpy as np
from mir_eval.transcription import precision_recall_f1_overlap as prf
from externals.Transkun.transkun.transcribe import readAudio, writeMidi
import collections
import copy
from externals.MIDI2ScoreTransformer.midi2scoretransformer.utils import infer
from externals.MIDI2ScoreTransformer.midi2scoretransformer.tokenizer import MultistreamTokenizer
from externals.MIDI2ScoreTransformer.midi2scoretransformer.models.roformer import Roformer
from externals.MIDI2ScoreTransformer.midi2scoretransformer.score_utils import postprocess_score

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
    mxl.write('musicxml', fp=f'{output_path}', makeNotation=True)

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


def oaf_pedal_extend(midi, CC_SUSTAIN=64):
        """
            Extend the sustain events in the MIDI file.
            
            Args:
                CC_SUSTAIN (int): MIDI control change number for sustain pedal. Default is 64.
            
            Returns:
                midi_copy (pretty_midi.PrettyMIDI): MIDI file with extended sustain events.
        """
        # make a copy of the MIDI file
        midi_copy = copy.deepcopy(midi)
        PEDAL_DOWN = 0
        PEDAL_UP = 1
        ONSET = 2
        OFFSET = 3

        # we will store all pedal and note events in a list
        # it will be sorted by time (pedal down, pedal up, onset, offset)
        events = []
        events.extend([(note.start, ONSET, [note, instrument]) for \
                       instrument in midi_copy.instruments for note in instrument.notes])
        events.extend([(note.end, OFFSET, [note, instrument]) for \
                       instrument in midi_copy.instruments for note in instrument.notes])
        
        for instrument in midi_copy.instruments:
            if not instrument.is_drum:
                for cc in instrument.control_changes:
                    if cc.number == CC_SUSTAIN:
                        if cc.value >= 64:
                            events.append((cc.time, PEDAL_DOWN, [cc, instrument]))
                        else:
                            events.append((cc.time, PEDAL_UP, [cc, instrument]))

        # sort the events by time and event type
        events.sort(key=lambda x: (x[0], x[1]))

        # We will keep a track of notes to extend (notes that fall within (pedal_down, pedal_up))
        # for each instrument
        extend_insts = collections.defaultdict(list)
        sus_insts = collections.defaultdict(bool) # stores sustain state for each instrument

        # We go through the events and extend the notes
        time = 0
        for time, event_type, event in events:
            if event_type == PEDAL_DOWN:
                sus_insts[event[1]] = True
            elif event_type == PEDAL_UP:
                sus_insts[event[1]] = False

                # If the pedal is up, we will end all of the notes
                # currently being extended
                ext_notes_inst = [] # Store new notes to extend
                for note in extend_insts[event[1]]:
                    if note.end < time:
                        # This note was extended, so we can end it
                        note.end = time
                    else:
                        # This note has not ended, so we still keep it
                        ext_notes_inst.append(note)
                extend_insts[event[1]] = ext_notes_inst
            elif event_type == ONSET:
                if sus_insts[event[1]]:
                    ext_notes_inst = []
                    # if sustain is on, we have to stop notes that are currently being extended
                    for note in extend_insts[event[1]]:
                        if note.pitch == event[0].pitch:
                            note.end = time
                            if note.start == note.end:
                                # it means this note now has zero duration,
                                # according to the official implementation, we should not keep it
                                event[1].notes.remove(note)
                        else:
                            # if the note is not the same as the one being extended,
                            # we can just add it to the list of notes to extend
                            ext_notes_inst.append(note)
                    extend_insts[event[1]] = ext_notes_inst
                
                # Add the new set of notes to extend to the list
                extend_insts[event[1]].append(event[0])
            elif event_type == OFFSET:
                # if susain is on, we will not end the note
                # so let's consider where sustain is off
                if not sus_insts[event[1]]:
                    if event[0] in extend_insts[event[1]]:
                        extend_insts[event[1]].remove(event[0])
            else:
                raise AssertionError(f"Unknown event type: {event_type}")    
        
        # End notes that are still being extended
        for instrument in extend_insts.values():
            for note in instrument:
                note.end = time
        
        # save this MIDI file (debugging purposes)
        #midi_copy.write("test.mid")
        return midi_copy

class Message:
    def __init__(self, time, type, note, velocity=0):
        """
            Initializes a MIDI message. This is an utility class
            to help with MIDI message processing using pretty_midi
            instead of MIDO used in the hfTransformer implementation
            by Sony.

            Args:
                time (float): The time of the message in seconds.
                type (str): The type of the message (e.g., 'note_on', 'note_off').
                note (int): The MIDI note number.
                velocity (int, optional): The velocity of the note. Defaults to 0.
        """
        self.time = time
        self.type = type
        self.note = note
        self.velocity = velocity
    
    def __str__(self):
        return f"Message(time={self.time}, type={self.type}, note={self.note}, velocity={self.velocity})"

    def __repr__(self):
        return self.__str__()

def get_notes(midi_obj: pretty_midi.PrettyMIDI):
    """
        Retrieve notes from a PrettyMIDI object and
        returns as a list of note dictionaries.

        Args:
            midi_obj (pretty_midi.PrettyMIDI): The PrettyMIDI object to extract notes from.

        Returns:
            notes (list): List of note dictionaries with keys 'onset', 'offset', 'pitch', 'velocity', 'reonset'.
    """
    notes = []
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append({
                    'onset': note.start,
                    'offset': note.end,
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'reonset': False  # Placeholder, reonset handling can be added if needed
                })
    
    # Sort notes by pitch and onset time
    notes.sort(key=lambda x: x['pitch']) # sort by pitch
    notes.sort(key=lambda x: x['onset']) # sort by onset time
    return notes

def extend_note_offsets(events, config: dict) -> list:
    """ 
        This method sort of mirrors _midi2note from hfTransformer implementation by Sony.
        However, the events represent a list of MIDI messages created using the Message class.
        The main purpose is to make the logic clearer. Extending note offsets can be simple
        yet tricky. Below is a detailed explanation of the logic.

        We have four types of events: 'note_on', 'note_off', 
        'control_change_on', 'control_change_off'.

        If the pedal is pressed (control_change_on), notes that are 
        ACTIVE should be extended.

        If the pedal is released (control_change_off), notes that are
        SUSTAINED and NOT ACTIVE should be ended. Because if they are
        ACTIVE, we should allow them continue...

        If it is a 'note_on' event, we mark the note as ACTIVE if it was
        neither ACTIVE nor SUSTAINED. Otherwise, it is a re-onset.

        If it is a 'note_off' event, we mark the note as not ACTIVE. If
        the pedal is pressed, we leave it as SUSTAINED.

        Args:
            events (list): List of MIDI events sorted by time.
            config (dict): Configuration dictionary containing useful info.

        Returns:
            notes (dict): Dictionary containing note information.
    """
    notes = [] # list containing note events ('onset', 'offset', 'pitch', 'velocity', 'reonset')
    active_notes = [False for _ in range(128)]
    sustained_notes = [False for _ in range(128)]
    reonset_notes = [False for _ in range(128)]
    onset_notes = [-1 for _ in range(128)]
    velocity_notes = [-1 for _ in range(128)]
    min_pitch = config['midi']['note_min']
    max_pitch = config['midi']['note_max']

    for i, event in enumerate(events):
        time = event.time.item()

        if event.type == 'control_change_off':
            # Sustain is off, so end all sustained notes that are not active
            for pitch in range(min_pitch, max_pitch + 1):
                if sustained_notes[pitch] and not active_notes[pitch]:
                    notes.append({
                        'onset': onset_notes[pitch],
                        'offset': time,
                        'pitch': pitch,
                        'velocity': velocity_notes[pitch],
                        'reonset': reonset_notes[pitch]
                    })
                    onset_notes[pitch] = -1
                    velocity_notes[pitch] = -1
                    reonset_notes[pitch] = False
            sustained_notes = [False for _ in range(128)]
        elif event.type == 'control_change_on':
            # Sustain is on, so mark all active notes as sustained
            for pitch in range(min_pitch, max_pitch + 1):
                if active_notes[pitch]:
                    sustained_notes[pitch] = True
        elif event.type == 'note_on':
            # Note on event
            # Check if it's a re-onset
            if active_notes[event.note] or sustained_notes[event.note]:
                # Re-onset
                notes.append({
                    'onset': onset_notes[event.note],
                    'offset': time,
                    'pitch': event.note,
                    'velocity': velocity_notes[event.note],
                    'reonset': reonset_notes[event.note]
                })
                reonset_notes[event.note] = True
            else:
                reonset_notes[event.note] = False
            onset_notes[event.note] = time
            velocity_notes[event.note] = event.velocity
            active_notes[event.note] = True
            if sustained_notes[event.note]:
                sustained_notes[event.note] = True
        elif event.type == 'note_off':
            # Note off event
            # End ONLY notes that are active and not sustained
            if active_notes[event.note] and not sustained_notes[event.note]:
                notes.append({
                    'onset': onset_notes[event.note],
                    'offset': time,
                    'pitch': event.note,
                    'velocity': velocity_notes[event.note],
                    'reonset': reonset_notes[event.note]
                })
                onset_notes[event.note] = -1
                velocity_notes[event.note] = -1
                reonset_notes[event.note] = False
            active_notes[event.note] = False

    # Handle any remaining active or sustained notes at the end
    final_time = events[-1].time
    for pitch in range(min_pitch, max_pitch + 1):
        if active_notes[pitch] or sustained_notes[pitch]:
            notes.append({
                'onset': onset_notes[pitch],
                'offset': final_time,
                'pitch': pitch,
                'velocity': velocity_notes[pitch],
                'reonset': reonset_notes[pitch]
            })

    # Perform sorting operations
    notes.sort(key=lambda x: x['pitch']) # sort by pitch
    notes.sort(key=lambda x: x['onset']) # sort by onset time
    return notes

def midi2note_hft(f_midi, extend_flag=True):
    # Get the midi object
    config = {"midi": {"note_min": 21, "note_max": 108}}
    midi_obj = pretty_midi.PrettyMIDI(f_midi)
    events = []

    if not extend_flag:
        # If we are not extending pedal, we can directly get the notes
        return get_notes(midi_obj)

    # store note events
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                msg_note_on = Message(time=note.start, type='note_on', note=note.pitch, velocity=note.velocity)
                msg_note_off = Message(time=note.end, type='note_off', note=note.pitch, velocity=0)
                events.append(msg_note_on)
                events.append(msg_note_off)

    # store control change events for sustain pedal
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for cc in instrument.control_changes:
                if cc.number == 64:  # Sustain pedal
                    value = cc.value
                    time = cc.time
                    msg_type = 'control_change_on' if value >= 64 else 'control_change_off'
                    msg_cc = Message(time=time, type=msg_type, note=None, velocity=value)
                    events.append(msg_cc)
    
    # For this to work correctly, we need to sort the events by time
    events.sort(key=lambda x: x.time)

    # Now, extend the note offsets based on the pedal events
    notes = extend_note_offsets(events, config)
    return notes

def get_scores_hft(pred_midi, gt_midi_file, extend_flag=True):
    if isinstance(pred_midi, str):
        pred_midi = pretty_midi.PrettyMIDI(pred_midi)
    else:
        pred_midi = pred_midi
        
    gt_midi_notes = midi2note_hft(gt_midi_file, extend_flag=extend_flag)
    gt_midi_list = []
    for each in gt_midi_notes:
        onset = each['onset']
        offset = each['offset']
        pitch = each['pitch']
        velocity = each['velocity']
        gt_midi_list.append([onset, offset, pitch, velocity])

    # get notes for predicted midi
    pred_midi_notes = []
    for instrument in pred_midi.instruments:
        for note in instrument.notes:
            pred_midi_notes.append((note.start, note.end, note.pitch, note.velocity))
    
    pred_midi_arr = np.array(pred_midi_notes)
    gt_midi_arr = np.array(gt_midi_list)
    pred_ints = pred_midi_arr[:, :2]
    pred_pitches = pred_midi_arr[:, 2]
    gt_ints = gt_midi_arr[:, :2]
    gt_pitches = gt_midi_arr[:, 2]
    p, r, f, _ = prf(gt_ints, gt_pitches, pred_ints, pred_pitches, offset_ratio=None)
    p_off, r_off, f_off, _ = prf(gt_ints, gt_pitches, pred_ints, pred_pitches)
    return f, f_off

def get_scores_oaf(pred_midi, gt_midi, extend_flag=True):
    # load ground truth midi
    if isinstance(gt_midi, str):
        gt_midi = pretty_midi.PrettyMIDI(gt_midi)
    else:
        gt_midi = gt_midi

    if isinstance(pred_midi, str):
        pred_midi = pretty_midi.PrettyMIDI(pred_midi)
    else:
        pred_midi = pred_midi
        
    # extend the ground truth midi using oaf's pedal extend function
    if extend_flag:
        gt_midi = oaf_pedal_extend(gt_midi)

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
    return f, f_off

def get_scores_kong(pred_midi, gt_midi, extend_flag=True):
    # load ground truth midi
    if isinstance(gt_midi, str):
        gt_midi = pretty_midi.PrettyMIDI(gt_midi)
    else:
        gt_midi = gt_midi

    if isinstance(pred_midi, str):
        pred_midi = pretty_midi.PrettyMIDI(pred_midi)
    else:
        pred_midi = pred_midi

    # extend the ground truth midi using OAF's pedal extend function
    # instead of Kong's buggy implementation

    #gt_midi = extend_pedal_kong(gt_midi)
    if extend_flag:
        gt_midi = oaf_pedal_extend(gt_midi)
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
    return f, f_off

def get_scores_trans(pred_midi, gt_midi, extend_flag=True):
    # load ground truth midi
    if isinstance(gt_midi, str):
        gt_midi = pretty_midi.PrettyMIDI(gt_midi)
    else:
        gt_midi = gt_midi

    if isinstance(pred_midi, str):
        pred_midi = pretty_midi.PrettyMIDI(pred_midi)
    else:
        pred_midi = pred_midi

    if extend_flag:
        gt_midi = oaf_pedal_extend(gt_midi)

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
    return f, f_off
