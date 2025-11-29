"""
    Filename: dataset_utils.py
    Description: Contains functionalities for dealing with data.
"""

# Import necessary libraries
import music21 as m21
import pretty_midi
import numpy as np

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

