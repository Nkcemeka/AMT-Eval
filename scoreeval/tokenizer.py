# MusicXML tokenizer to compute word error rate
# Credits: https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/blob/main/midi2scoretransformer/tokenizer.py
# Import the necessary files
import math
from fractions import Fraction
from typing import Dict, List
import pretty_midi
import torch
import torch.nn.functional as F
from music21 import (
    articulations,
    clef,
    converter,
    expressions,
    instrument,
    key,
    meter,
    note,
    stream,
    tempo,
)
from music21.common.numberTools import opFrac
from music21.midi.translate import prepareStreamForMidi

class Downbeat:
    MEASURE_NUMBER = 0
    OFFSET = 1
    LAST_OFFSET = 2
    MEASURE_LENGTH = 3


db_config = Downbeat.MEASURE_LENGTH

PARAMS = {
    "offset": {"min": 0, "max": 6, "step_size": 1 / 24},
    "duration": {"min": 0, "max": 4, "step_size": 1 / 24},
    "downbeat": {"min": -1 / 24, "max": 6, "step_size": 1 / 24},
}

def realize_spanners(s):
    to_remove = []
    for sp in s.recurse().getElementsByClass(expressions.TremoloSpanner):
        l = sp.getSpannedElements()
        if len(l) != 2:
            print("Not sure what to do with this spanner", sp, l)
            continue
        start, end = l

        offset = start.offset
        start_chord = None
        end_chord = None
        startActiveSite = start.activeSite
        endActiveSite = end.activeSite
        if start.activeSite is None:
            start_chord: chord.Chord = start._chordAttached
            offset = start_chord.offset
            startActiveSite = start._chordAttached.activeSite
        if end.activeSite is None:
            end_chord: chord.Chord = end._chordAttached
            endActiveSite = end._chordAttached.activeSite

        # We insert a tremolo expression on the start note
        # realize it, and then change every second note to have the pitch of the end note
        trem = expressions.Tremolo()
        trem.measured = sp.measured
        trem.numberOfMarks = sp.numberOfMarks
        start.expressions.append(trem)
        out = trem.realize(start, inPlace=True)[0]
        if start_chord:
            if len(start_chord.notes) == 1:
                startActiveSite.remove(start_chord)
            else:
                start_chord.remove(start)
        else:
            startActiveSite.remove(start)
        if end_chord:
            if len(end_chord.notes) == 1:
                endActiveSite.remove(end_chord)
            else:
                end_chord.remove(end)
        else:
            endActiveSite.remove(end)
        for i, n2 in enumerate(out):
            if i % 2 == 1:
                n2.pitch = end.pitch
            startActiveSite.insert(offset, n2)
            offset += n2.duration.quarterLength
        to_remove.append(sp)
    for sp in s.recurse().getElementsByClass(expressions.TrillExtension):
        l = sp.getSpannedElements()
        start = l[0]
        exp = [l.expressions for l in l]
        if not any(isinstance(e, expressions.Trill) for ex in exp for e in ex):
            if len(l) != 1:
                print("Not sure what to do with this spanner", sp, l)
                continue
            start.expressions.append(expressions.Trill())
            to_remove.append(sp)
    s.remove(to_remove, recurse=True)
    return s


def one_hot_bucketing(
    values: torch.Tensor | List[int | float], min, max, buckets=None, step_size=None
) -> torch.Tensor:
    assert buckets is not None or step_size is not None
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    if values.ndim == 2:
        values = values.squeeze(1)
    values = values.float()

    # discretize the values into buckets
    if buckets is None:
        buckets = int((max + step_size - min) / step_size)
        bucket_indices = ((values - min) / (max + step_size - min) * buckets).round()
    else:
        bucket_indices = (values - min) / (max - min) * buckets
    # clamp the bucket indices to be between 0 and n_buckets - 1
    bucket_indices = bucket_indices.long().clamp(0, buckets - 1)
    one_hots = F.one_hot(bucket_indices, num_classes=buckets)
    return one_hots


class MultistreamTokenizer:
    @staticmethod
    def mxl_to_list(mxl_path: str) -> tuple[List[note.Note], stream.Score]:
        """Converts a music21 stream to a sorted and deduplicated list of notes.

        Parameters
        ----------
            mxl_path : str
                Path to the musicxml file.

        Returns
        -------
            List[music21.note.Note]:
                The list of notes in the music21 stream.
            music21.stream.Score:
                The music21 stream. This is only returned to
                ensure that the stream is not garbage collected.
        """
        mxl = converter.parse(mxl_path, forceSource=True)
        mxl = realize_spanners(mxl)
        mxl: stream.Score = mxl.expandRepeats()
        # strip all ties inPlace
        mxl.stripTies(preserveVoices=False, inPlace=True)
        # Realize Tremolos
        for n in mxl.recurse().notes:
            for e in n.expressions:
                if isinstance(e, expressions.Tremolo):
                    offset = n.offset
                    out = e.realize(n, inPlace=True)[0]
                    v = n.activeSite
                    v.remove(n)
                    for n2 in out:
                        v.insert(offset, n2)
                        offset += n2.duration.quarterLength
                    break
        mxl = prepareStreamForMidi(mxl)

        notes: list[note.Note] = []
        assert not any(note.isChord for note in mxl.flatten().notes)

        for n in mxl.flatten().notes:
            # if note.style.noteSize == "cue":
            #     continue
            if n.style.hideObjectOnPrint:
                continue
            n.volume.velocity = int(round(n.volume.cachedRealized * 127))
            notes.append(n)
        # Sort like this to preserve correct order for grace notes.
        def sortTuple(n):
           # Sort by offset, then pitch, then duration
           # Grace notes that share the same offset are sorted by their insertIndex
           # instead of their pitch as they rarely actually occur simultaneously
           return (
               n.offset,
               not n.duration.isGrace,
               n.pitch.midi if not n.duration.isGrace else n.sortTuple(mxl).insertIndex,
               n.duration.quarterLength
           )
        #    return (n.offset, n.pitch.midi, n.duration.quarterLength)
        notes_sorted = sorted(notes, key=sortTuple)
        notes_consolidated: list[note.Note] = []
        last_note = None
        for n in notes_sorted:
            if last_note is None or n.offset != last_note.offset or n.pitch.midi != last_note.pitch.midi:
                notes_consolidated.append(n)
                last_note = n
            elif last_note.duration.isGrace:
                last_note = n
            else:
                if n.duration.quarterLength > last_note.duration.quarterLength:
                    last_note = n
        # sort again because we might have changed the duration of grace notes
        notes_consolidated = sorted(notes_consolidated, key=sortTuple)
        return notes_consolidated, mxl

    @staticmethod
    def parse_mxl(mxl_path) -> Dict[str, torch.Tensor]:
        """
        Converts a MusixXML file to a list of tensors.
        All tensors have shape (n_notes,) and no quantization is applied yet.
        Used during preprocessing.

        Parameters
        ----------
        mxl_path : str
            Path to the musicxml file.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dict of tensors of shape (n_notes, 1) with keys "offset"
            "downbeat", "duration", "pitch", "accidental", "velocity", "grace", "trill",
            "staccato", "voice", "stem", "hand".
        """
        # return mxl_stream for garbage collection reasons only
        mxl_list, mxl_stream = MultistreamTokenizer.mxl_to_list(mxl_path)
        if len(mxl_list) == 0:
            offset_stream = torch.Tensor([])
            downbeat_stream = torch.Tensor([])
            duration_stream = torch.Tensor([])
            pitch_stream = torch.Tensor([])
            accidental_stream = torch.Tensor([])
            keysignature_stream = torch.Tensor([])
            velocity_stream = torch.Tensor([])
            grace_stream = torch.Tensor([])
            trill_stream = torch.Tensor([])
            staccato_stream = torch.Tensor([])
            voice_stream = torch.Tensor([])
            stem_stream = torch.Tensor([])
            hand_stream = torch.Tensor([])
        else:
            # fmt: off
            note_offsets = torch.FloatTensor([n.offset for n in mxl_list])
            measure_offsets = torch.FloatTensor([n.getContextByClass("Measure").offset for n in mxl_list])
            offset_stream = note_offsets - measure_offsets

            if db_config == Downbeat.MEASURE_NUMBER:
                nums = torch.tensor([n.getContextByClass("Measure").number for n in mxl_list])
                downbeat_stream = (torch.diff(nums, prepend=torch.tensor([1])) > 0).float()
            elif db_config == Downbeat.OFFSET:
                downbeat_stream = torch.logical_or(offset_stream == 0, torch.diff(offset_stream, prepend=torch.tensor([0.0])) < 0).float()
            elif db_config == Downbeat.LAST_OFFSET:
                downbeat_stream = torch.diff(measure_offsets, prepend=torch.tensor([0.0])) > 0
                shifts = measure_offsets - torch.cat((torch.tensor([0]), note_offsets[:-1]))
                downbeat_stream = torch.where(downbeat_stream, shifts, torch.ones_like(downbeat_stream).float() * PARAMS["downbeat"]["min"])
            elif db_config == Downbeat.MEASURE_LENGTH:
                downbeat_stream = torch.diff(measure_offsets, prepend=torch.tensor([0.0]))
                downbeat_stream[downbeat_stream<=0] = PARAMS["downbeat"]["min"]

            duration_stream = torch.Tensor([n.duration.quarterLength for n in mxl_list])
            pitch_stream = torch.Tensor([n.pitch.midi for n in mxl_list])
            velocity_stream = torch.Tensor([n.volume.velocity for n in mxl_list])
            def alter_map(accidental):
                if accidental is None:
                    return 5
                alter_to_value = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
                # if not in the mapping, return 6 (for unknown)
                return alter_to_value.get(accidental.alter, 6)
            accidental_stream = torch.Tensor([alter_map(n.pitch.accidental) for n in mxl_list])
            # for each note offset, find the last key that occurs before or at the same time as it
            keysignatures = {float(e.offset): e for e in mxl_stream.flatten().getElementsByClass(key.KeySignature)}
            keysignature_stream = torch.Tensor([next(((v.sharps if v.sharps is not None else 8) for k, v in reversed(keysignatures.items()) if k <= n), 8) for n in note_offsets]) + 7
            # MusicXML attribute streams
            grace_stream = torch.Tensor([n.duration.isGrace for n in mxl_list])
            trills = (expressions.Trill, expressions.InvertedMordent, expressions.Mordent, expressions.Turn)
            trill_stream = torch.Tensor([any(isinstance(e, trills) for e in n.expressions) for n in mxl_list])
            staccatos = (articulations.Staccatissimo, articulations.Staccato)
            staccato_stream = torch.Tensor([any(isinstance(e, staccatos) for e in n.articulations) for n in mxl_list])
            voices = [n.getContextByClass("Voice") for n in mxl_list]
            voice_stream = torch.Tensor([int(v.id) if v is not None else 0 for v in voices])
            stem_map = {"up": 0, "down": 1, "noStem": 2}
            stem_stream = torch.Tensor([stem_map.get(n.stemDirection, 3) for n in mxl_list])
            # fmt: on
            # Hands/Staff logic is slightly more complicated
            #
            hand_stream = []
            not_matched = set()
            for n in mxl_list:
                # Usually part names are similar to "[P1-Staff2]"
                # Added str below because my extracted xml files might not have a string
                part_name = str(n.getContextByClass("Part").id).lower()
                if "staff1" in part_name:
                    hand_stream.append(0)
                elif "staff2" in part_name:
                    hand_stream.append(1)
                else:
                    hand_stream.append(2)
                    if part_name not in not_matched:  # only one warning per part
                        not_matched.add(part_name)
                        # print("Couldn't match", part_name)
            hand_stream = torch.tensor(hand_stream)
        mxl_stream  # keep stream for gc only
        return {
            "offset": offset_stream,
            "downbeat": downbeat_stream,
            "duration": duration_stream,
            "pitch": pitch_stream,
            "accidental": accidental_stream,
            "keysignature": keysignature_stream,
            "velocity": velocity_stream,
            "grace": grace_stream,
            "trill": trill_stream,
            "staccato": staccato_stream,
            "voice": voice_stream,
            "stem": stem_stream,
            "hand": hand_stream,
        }

    @staticmethod
    def bucket_mxl(mxl_streams: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Bucketing TODO: checkout bucketing
        # fmt: off
        offset_stream = one_hot_bucketing(mxl_streams["offset"], **PARAMS["offset"])
        duration_stream = one_hot_bucketing(mxl_streams["duration"], **PARAMS["duration"])
        downbeat_stream = one_hot_bucketing(mxl_streams["downbeat"], **PARAMS["downbeat"])
        pitch_stream = one_hot_bucketing(mxl_streams["pitch"], 0, 127, 128)
        accidental_stream = one_hot_bucketing(mxl_streams["accidental"], 0, 6, 7)
        keysignature_stream = one_hot_bucketing(mxl_streams["keysignature"], 0, 15, 16)
        velocity_stream = one_hot_bucketing(mxl_streams["velocity"], 0, 127, 8)
        grace_stream = one_hot_bucketing(mxl_streams["grace"], 0, 1, 2)
        trill_stream = one_hot_bucketing(mxl_streams["trill"], 0, 1, 2)
        staccato_stream = one_hot_bucketing(mxl_streams["staccato"], 0, 1, 2)
        voice_stream = one_hot_bucketing(mxl_streams["voice"], 0, 8, 9)
        stem_stream = one_hot_bucketing(mxl_streams["stem"], 0, 3, 4)
        hand_stream = one_hot_bucketing(mxl_streams["hand"], 0, 2, 3)
        # fmt: on
        # Beams
        # Slurs
        # Tuplets
        # Dots?
        return {
            "offset": offset_stream.float(),
            "downbeat": downbeat_stream.float(),
            "duration": duration_stream.float(),
            "pitch": pitch_stream.float(),
            "accidental": accidental_stream.float(),
            "keysignature": keysignature_stream.float(),
            "velocity": velocity_stream.float(),
            "grace": grace_stream.float(),
            "trill": trill_stream.float(),
            "staccato": staccato_stream.float(),
            "voice": voice_stream.float(),
            "stem": stem_stream.float(),
            "hand": hand_stream.float(),
           # "pad": torch.ones((offset_stream.shape[0],), dtype=torch.long), # Not necessary for metric evaluation
        }

    @staticmethod
    def tokenize_mxl(mxl_path: str) -> Dict[str, torch.Tensor]:
        """Converts a MusicXML file to a list of tensors of shape (n_notes).

        Parameters
        ----------
        mxl_path : str
            Path to the musicxml file.

        Returns
        -------
        torch.Tensor
            returns a list of tensors of shape (n_notes,)
        """
        mxl_streams = MultistreamTokenizer.parse_mxl(mxl_path)
        return MultistreamTokenizer.bucket_mxl(mxl_streams)
