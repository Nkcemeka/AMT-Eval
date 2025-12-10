import librosa
import argparse
import csv
import json
from pathlib import Path
import soundfile as sf
import music21 as m21
import pretty_midi
import pandas as pd
from utils import get_midi_note_events_strict, get_tempo_changes, get_ksig_changes, \
            get_tsig_changes, get_xml_score, remove_broken_ties

class ASAPDataset:
    """
        Works for all datasets that follow the
        ASAP Dataset structure.
    """
    def __init__(self, asap_path: str="/home/nkcemeka/Documents/Datasets/ASAP",
                 num_measures=2, store_dir: str="./data", test_split: str="maestro") -> None:
        """
            Instantiate Dataset class!

            Args:
            -----
                asap_path (str): Base path to asap dataset
                num_measures (int): Number of measures to extract
                store_dir (str): Directory to store data
                test_split (str): Use "maestro" or "beyer's" test split. Default is MAESTRO.
        """
        self.asap_path = asap_path
        self.asap_annots = json.load(open(Path(self.asap_path) / "asap-dataset/asap_annotations.json"))
        self.ts = test_split
        _, _, test = self.get_train_val_test_asap()

        for test_sample in test:
            filename = test_sample[1]
            print(f"Processing {filename}...")
            self.window_view(filename, num_measures=num_measures, store_dir=store_dir)

    def map_test_to_maestro(self, maestro_path: str, ext_midi: str,\
            ext_audio: str, maestro_csv="maestro-v2.0.0.csv") -> dict:
        """
            Gets a mapping of piece_id from ACPAS metadata_R.csv 
            dataset and the test set audio files in MAESTRO-V2
            
            Args:
            ------
                maestro_path (str): Base path to maestro
                ext_midi (str): MIDI extension
                ext_audio (str): Audio extension
                maestro_csv (str): Metadata name for maestro
            
            Returns:
            ---------
                result_map (dict): Mapping of piece_id to test files in 
                                   maestro
        """
        _, _, test = self.get_maestro_train_val_test(maestro_path, ext_midi, ext_audio, csv_name=maestro_csv)
        # Get the test audio files from test
        test_audio = [str(each[0]) for each in test]

        meta_asap = pd.read_csv(Path(self.asap_path)/"asap-dataset/metadata.csv")
        meta_r = Path(self.asap_path) / "ACPAS/metadata_R.csv"
        meta_r_csv = pd.read_csv(meta_r)
        ids = []
        subset = meta_r_csv.loc[:, ["piece_id", "performance_MIDI_external"]]
        subset_perf = subset["performance_MIDI_external"]
        subset_piece = subset["piece_id"]

        for idx, item in enumerate(subset_perf):
            split = item.split("/")
            if split[0] != "{ASAP}":
                continue
            name = str(Path(split[-1]).stem)
            split.pop(0) # remove {ASAP}
            split.pop() # Remove the MIDI name
            id = '/'.join(split) + f"/{name}.wav"
            ids.append((idx, id))

        # Parse the test files 
        count = 0
        result_map = {} # (piece_idx in metadata_R to maestro file path)
        for (idx, each_id) in ids:
            bool_idx = meta_asap["audio_performance"] == each_id
            if meta_asap["maestro_audio_performance"][bool_idx].item().replace("{maestro}", \
                    maestro_path) in test_audio:
                # Note that two different pieces can have the same piece_id 
                # But I don't care about that for now
                value = meta_asap["maestro_audio_performance"][bool_idx].item()
                result_map[subset_piece.iloc[idx].item()] = value

        return result_map

    def get_maestro_train_val_test(self, path, ext_midi, ext_audio, csv_name="maestro-v2.0.0.csv"):
        """
            Get the train, validation, test split for MAESTRO!
            This can be important in using the test split of MAESTRO
            for ASAP.

            Args:
            -----
                path (str): Base path to MAESTRO dataset
                ext_midi (str): MIDI extension
                ext_audio (str): Audio extension
        """
        train_files = []
        val_files = []
        test_files = []

        # metadata_csv is structured as follows:
        # canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration
        # read the csv file
        metadata_csv = Path(path) / f"{csv_name}"
        assert metadata_csv.exists(), f"{metadata_csv} does not exist"
        
        with open(metadata_csv, 'r') as f:
            content = csv.reader(f, delimiter=',', quotechar='"')

            base_path = Path(path)
            next(content)  # skip the header

            for i, each in enumerate(content):
                if each[2] == 'train':
                    midi_path = base_path / each[4]
                    audio_path = str(base_path / each[4]).replace(f".{ext_midi}", f".{ext_audio}")
                    audio_path = Path(audio_path)
                    train_files.append((audio_path, midi_path))
                    assert audio_path.exists(), f"{audio_path} does not exist"
                    assert midi_path.exists(), f"{midi_path} does not exist"
                elif each[2] == 'validation':
                    midi_path = base_path / each[4]
                    audio_path = str(base_path / each[4]).replace(f".{ext_midi}", f".{ext_audio}")
                    audio_path = Path(audio_path)
                    val_files.append((audio_path, midi_path))
                    assert audio_path.exists(), f"{audio_path} does not exist"
                    assert midi_path.exists(), f"{midi_path} does not exist"
                elif each[2] == 'test':
                    midi_path = base_path / each[4]
                    audio_path = str(base_path / each[4]).replace(f".{ext_midi}", f".{ext_audio}")
                    audio_path = Path(audio_path)
                    test_files.append((audio_path, midi_path))
                    assert audio_path.exists(), f"{audio_path} does not exist"
                    assert midi_path.exists(), f"{midi_path} does not exist"
                else:
                    raise ValueError(f"Split {each[2]} not supported")
        return train_files, val_files, test_files

    def gen_midi_from_notes(self, framed_midi: list) -> pretty_midi.PrettyMIDI:
        """
            Generate MIDI file from list of note events.

            Args:
            -----
                framed_midi (list): List of note events for a frame which
                                    can be a couple of measures long

            Returns:
            --------
                framed_midi_obj (pretty_midi.PrettyMIDI): PrettyMIDI object
        """
        framed_midi_obj = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        for event in framed_midi:
            note = pretty_midi.Note(velocity=int(event[3]), pitch=int(event[2]), start=event[0], end=event[1])
            piano.notes.append(note)
        framed_midi_obj.instruments.append(piano)
        return framed_midi_obj
    
    def bsearch_robust(self, x: list, target) -> int:
        """
            A more robust binary search that returns the index 
            of the element that is equal or just exceeds the target.

            Args:
            -----
                x (list): sorted list
                target (float | int): Target to search for

            Returns:
            --------
                lp (int): Index of element that equals or exceeds target. 
        """
        lp = 0
        rp = len(x) - 1

        while lp <= rp:
            mid = (lp+rp)//2

            if x[mid] == target:
                return mid
            elif x[mid] < target:
                lp = mid + 1
            else:
                rp = mid - 1
        
        return lp

    def bsearch(self, x: list, target) -> int:
        """
            We are guranteed to find our target so ideally, we
            should not get -1. Time complexity of this algorithm is O(log n)

            Args:
            -----
                x (list): sorted list
                target (float | int): : target of interest

            Returns:
            -------
                mid (int): Returns index of target and -1 if target is 
                       not found.
        """

        lp = 0
        rp = len(x) - 1

        while lp <= rp:
            mid = (lp+rp)//2

            if x[mid] == target:
                return mid
            elif target < x[mid]:
                # Check the left part of the array
                rp = mid - 1
            else:
                # Check the right part of the array
                lp = mid + 1
        
        return -1
    
    def get_key_sig(self, key_sigs: dict, mscore_db: list) -> list:
        """
            Gets the key signature changes on a downbeat level.

            Args:
            -----
                key_sigs (dict): Dictionary of key signatures on a beat level
                                 with time as key and signature as value
                mscore_db (list): List of downbeats on a score level.

            Returns:
            --------
                key_sig_list (list): List of key signature changes on a downbeat
                                     level.
        """
        # We treat key signatures separately because these changes
        # occur on a beat level and not necessarily on a downbeat level
        # Time complexity is O(k log n) where k is number of key signature changes
        # Since k is typically very small, avg. time complexity is O(log n)
        key_sig_list = []
        prev_idx = None
        for time in key_sigs.keys():
            # Get the index of the beat that is >= time
            idx = self.bsearch_robust(mscore_db, float(time))

            if prev_idx is None and len(key_sigs.keys()) == 1:
                key_sig_list.append(key_sigs[time])
                key_sig_list.extend([key_sigs[time]] * (len(mscore_db) - 1))
            elif prev_idx is None and len(key_sigs.keys()) > 1:
                key_sig_list.append(key_sigs[time])
            else:
                key_sig_list.extend([key_sig_list[-1]] * (idx - prev_idx - 1))
                key_sig_list.append(key_sigs[time])

            prev_idx = idx
        
        # Fill the rest of the list with the last key signature
        if len(key_sig_list) < len(mscore_db):
            key_sig_list.extend([key_sig_list[-1]] * (len(mscore_db) - len(key_sig_list)))

        assert len(key_sig_list) <= len(mscore_db), f"Length mismatch in key signature list and midi score downbeats,\
              {len(key_sig_list)} vs {len(mscore_db)}"
        return key_sig_list

    def get_time_sig(self, sig: dict, mscore_db: list):
        """
            Gets the time signatures on a downbeat level.

            Args:
            -----
                sig (dict): Time signature dict. Time is key and 
                            signature is value.
                mscore_db (list): list of downbeats in the midi score.

            Returns:
            --------
                sig_list (list): Time signature list on a downbeat level.

        """
        # We could use an O(n) approach but that is not a good idea 
        # since time signature changes are rare events
        # so we use binary search to find the latest time signature
        # Time complexity is O(k log n) where k is number of time and signature changes
        # Since k is typically very small, avg. time complexity is O(log n)
        sig_list = []
        for i, key in enumerate(sig.keys()):
            idx = self.bsearch(mscore_db, float(key))
            if idx == -1:
                raise ValueError(f"Signature time {key} not found in midi score downbeats!")
            
            # Get the length of sig_list
            curr_len = len(sig_list)
            if curr_len == 0:
                sig_list.extend([None]*idx)
            else:
                sig_list.extend([sig_list[-1]]*(idx - curr_len))
            sig_list.append(sig[key])
        
        # Fill the rest of the list with the last key signature
        if len(sig_list) < len(mscore_db):
            sig_list.extend([sig_list[-1]]*(len(mscore_db) - len(sig_list)))
        
        # Check that lengths match
        assert len(sig_list) == len(mscore_db), "Length mismatch in signature list and midi score downbeats"
        return sig_list

    def window_view(self, filename: str, num_measures: int = 2, store_dir: str = "./data"):
        """
            Generates windowed segments for a file and stores them
            in the store_dir folder.

            Args:
            -----
                filename (str): Name of file
                num_measures (int): Number of measures to extract for each segment
                store_dir (str): Directory to store the data
        """
        # Create store directory if it does not exist
        Path(store_dir).mkdir(parents=True, exist_ok=True)

        # Create audio, midi, midi_score, xml_score directories in store_dir
        for subdir in ["audio", "midi", "midi_score", "xml_score"]:
            subdir_path = Path(store_dir) / subdir
            Path(subdir_path).mkdir(parents=True, exist_ok=True)

        annots = self.asap_annots[filename.replace(str(Path(self.asap_path) / "asap-dataset/") + "/", "")]
        perf_db = annots["performance_downbeats"] # performance downbeats in seconds
        mscore_db = annots["midi_score_downbeats"] # midi score downbeats in seconds
        score_map = annots["downbeats_score_map"] # maps midi score downbeats to XML measures
        key_sig = annots["midi_score_key_signatures"] # key signature of the piece
        time_sig = annots["midi_score_time_signatures"] # time signature of the piece

        # Time signature for the midi_score should occur on a downbeat level
        # This is an assumption but seems reasonable
        # For time signatures, I am pretty confident it is always on a downbeat
        time_sig_list = self.get_time_sig(time_sig, mscore_db)

        # for key signatures, it occurs on a beat level
        key_sig_list = self.get_key_sig(key_sig, mscore_db)

        # key is 'm{i}' where i is measure number
        # value is a dict with keys 'pstart' and 'pend' giving performance start and end times in seconds
        # 'sstart' and 'send' giving score start and end times in seconds
        # 'measure_number' giving the measure number in the score
        mhashmap = {} 
        assert len(perf_db) == len(mscore_db) == len(score_map), f"Length mismatch in downbeats and score map"
        num_db = len(perf_db)
        
        for i in range(0, num_db - num_measures, num_measures):
            # We will use a two pointer approach to build the hashmap
            j = i + num_measures - 1 # num_measures determines how much measures we consider in one window

            # init the measure entry
            if isinstance(score_map[i], str) or isinstance(score_map[j], str):
                # We don't want to deal with this
                # weird edge case
                continue
                
            mhashmap[f'm{i+1}'] = {} # m(db) for the key where db is downbeat index

            # measure number start
            mhashmap[f'm{i+1}']['measure_number_start'] = int(score_map[i]) + 1 # +1 to convert from 0-indexed to 1-indexed
            # measure number end
            mhashmap[f'm{i+1}']['measure_number_end'] = int(score_map[j]) + 1 # +1 to convert from 0-indexed to 1-indexed

            # performance start and end times
            mhashmap[f'm{i+1}']['pstart'] = float(perf_db[i])
            mhashmap[f'm{i+1}']['pend'] = float(perf_db[j+1])

            # score start and end times
            mhashmap[f'm{i+1}']['sstart'] = float(mscore_db[i])
            mhashmap[f'm{i+1}']['send'] = float(mscore_db[j+1])

            # Store key and time signatures between i and j
            for each in range(i, j+2):
                if 'key_signature' not in mhashmap[f'm{i+1}']:
                    mhashmap[f'm{i+1}']['key_signature'] = []
                mhashmap[f'm{i+1}']['key_signature'].append(key_sig_list[each])
            
                if 'time_signature' not in mhashmap[f'm{i+1}']:
                    mhashmap[f'm{i+1}']['time_signature'] = []
                mhashmap[f'm{i+1}']['time_signature'].append(time_sig_list[each])

        # Now that we are done building the hashmap, we generate the audio, midi, midi_score, and xml_score files
        audio_file = filename.replace(".mid", ".wav")
        midi_file = filename
        midi_score_file = Path(filename).parent / "midi_score.mid"
        xml_score_file = Path(filename).parent / "xml_score.musicxml"

        assert Path(audio_file).exists(), f"Audio file {audio_file} does not exist"
        assert Path(midi_file).exists(), f"MIDI file {midi_file} does not exist"
        assert Path(midi_score_file).exists(), f"MIDI score file {midi_score_file} does not exist"
        assert Path(xml_score_file).exists(), f"XML score file {xml_score_file} does not exist"

        audio = librosa.load(audio_file, sr=16000)[0]
        midi = pretty_midi.PrettyMIDI(str(midi_file))
        midi_score = pretty_midi.PrettyMIDI(str(midi_score_file))
        score_xml = m21.converter.parse(str(xml_score_file))

        for key in mhashmap.keys():
            measure_num_start = mhashmap[key]['measure_number_start']
            measure_num_end = mhashmap[key]['measure_number_end']
            measure_num_list = [measure_num_start, measure_num_end]

            pstart = mhashmap[key]['pstart']
            pend = mhashmap[key]['pend']
            sstart = mhashmap[key]['sstart']
            send = mhashmap[key]['send']
            ks = mhashmap[key]['key_signature']
            ts = mhashmap[key]['time_signature']

            # Get audio segment
            audio_start_sample = int(round(pstart * 16000))
            audio_end_sample = int(round(pend * 16000)) + 1
            audio_segment = audio[audio_start_sample:audio_end_sample]

            # Get MIDI segment
            midi_segment = get_midi_note_events_strict(midi, pstart, pend)
            midi_segment = self.gen_midi_from_notes(midi_segment)

            # Get MIDI score segment
            midi_score_segment = get_midi_note_events_strict(midi_score, sstart, send)
            midi_seg_tempo_changes = get_tempo_changes(midi_score, sstart, send)
            midi_seg_ts_changes = get_tsig_changes(midi_score, sstart, send)
            midi_seg_ks_changes = get_ksig_changes(midi_score, sstart, send)
            midi_score_segment = self.gen_midi_from_notes(midi_score_segment)
            midi_score_segment.resolution = midi_score.resolution # Update resolution

            # Add tempo changes, key signature changes and time signature changes to new MIDI segment
            tempo_time, tempi = midi_seg_tempo_changes

            last_tick = 0
            last_tick_scale = 60.0/(tempi[0].item() * midi_score_segment.resolution)
            previous_time = 0.
            midi_score_segment._tick_scales = [(last_tick, last_tick_scale)]

            for i in range(1, len(tempo_time)):
                # compute new tick position
                tick = last_tick + (tempo_time[i].item() - previous_time)/last_tick_scale 
                # Update tick scale
                tick_scale = 60.0 / (tempi[i].item() * midi_score_segment.resolution)
                # Don't add repeat tick scales
                if tick_scale != last_tick_scale:
                    midi_score_segment._tick_scales.append((int(round(tick)), tick_scale))
                    previous_time = tempo_time[i].item()
                    last_tick, last_tick_scale = tick, tick_scale

            # Update tick to time mapping
            midi_score_segment._update_tick_to_time(midi_score_segment._tick_scales[-1][0] + 1)
            midi_score_segment.time_signature_changes = midi_seg_ts_changes
            midi_score_segment.key_signature_changes = midi_seg_ks_changes

            # Get XML score segment
            score_xml_segment = get_xml_score(score_xml, measure_num_list, ks, ts)

            # Save the segments to files
            audio_store_path = Path(store_dir) / "audio" / f"{Path(filename).stem}_m{measure_num_start}.wav"
            midi_store_path = Path(store_dir) / "midi" / f"{Path(filename).stem}_m{measure_num_start}.mid"
            midi_score_store_path = Path(store_dir) / "midi_score" / f"{Path(filename).stem}_m{measure_num_start}.mid"
            xml_score_store_path = Path(store_dir) / "xml_score" / f"{Path(filename).stem}_m{measure_num_start}.musicxml"

            # Save audio
            try:
                # If the score saves to fail for some reason, then skip
                remove_broken_ties(score_xml_segment, str(xml_score_store_path)) # This function saves the file as well
                #score_xml_segment.write('musicxml', fp=str(xml_score_store_path), makeNotation=True)
            except:
                continue

            sf.write(str(audio_store_path), audio_segment, 16000)
            midi_segment.write(str(midi_store_path))
            midi_score_segment.write(str(midi_score_store_path))

    def get_train_val_test_asap(self, opts: bool = True) -> tuple[list, list, list]:
        """ 
            Parse the ASAP dataset and return train, val, test splits.
            Credits: `https://github.com/TimFelixBeyer/asap-dataset`

            Args:
                asap_path (str): Path to the ASAP dataset.
                opts (bool): Whether to remove items without performance audio.

            Returns:
                tuple[list, list, list]: train, val, test splits
        """
        # We are going to select five files from a dataset that has MIDI-XML pairs and compute the metrics on them.
        # We will write a function to parse the ASAP dataset and return a list of (audio, score-midi, p-midi, xml) tuples.
        ASAP_PATH = self.asap_path

        # load the metadata (we will use Tim Beyer's approach)
        real_data  = pd.read_csv(str(Path(ASAP_PATH) / "ACPAS/metadata_R.csv"))
        synth_data = pd.read_csv(str(Path(ASAP_PATH) / "ACPAS/metadata_S.csv"))
        asap_annots = self.asap_annots

        UNALIGNED = set(
            "{ASAP}/" + k 
            for k, v in asap_annots.items()
            if not v["score_and_performance_aligned"]
        )

        # SKIP contains files not parsable by music21
        SKIP = set(
            ["{ASAP}/Glinka/The_Lark/Denisova10M.mid", "{ASAP}/Glinka/The_Lark/Kleisen07M.mid"]
        )

        # Indices to ignore because of wrong alignment or other issues
        TO_IGNORE_INDICES = [152, 153, 154, 165, 166, 179, 180, 181, 332, 333, 334, 335, 349, 350,
                            351, 418, 419, 420, 426, 428, 429, 430, 472, 473, 474, 489, 490, 491,
                            516, 517, 518, 519, 520, 521, 522, 540, 541, 560, 609, 774, 798, 799,
                            800, 801, 802, 803, 819, 920, 921, 935, 936, 937, 938, 939, 940, 941,
                            979, 980, 981, 997, 998, 999, 1012, 1013, 1014, 1017, 1018]


        # To keep eval consistent, we hardcode test piece ids here.
        #TEST_PIECE_IDS = [15, 78, 159, 172, 254, 288, 322, 374, 395, 399, 411, 418, 452, 478]
        # The ones below are gotten from maestro-v2 dataset
        # I also remove piece IDS in ignore_indices and IDs that are problematic
        if self.ts == "maestro":
            TEST_PIECE_IDS = [20, 22, 23, 30, 41, 52, 54, 55, 62, 73, 78, 83, 90, 91, 95, 98, 121, \
                    122, 129, 136, 138, 139, 141, 223, 232, 255, 324, 325, 341, 342, 343, 401, \
                    421, 443, 445, 447]
        elif self.ts == "beyer":
            TEST_PIECE_IDS = [15, 78, 159, 172, 254, 288, 322, 374, 395, 399, 411, 418, 452, 478]
        else:
            raise ValueError("Unknown test split. Available options are: `maestro` and `beyer`")

        data = pd.concat([real_data, synth_data])

        # Perform filtering
        data = data[(data["source"] == "ASAP") & data["aligned"]]
        data = data[~data["performance_MIDI_external"].isin(SKIP)]
        data = data[~data["performance_MIDI_external"].isin(UNALIGNED)]
        data = data.drop_duplicates(subset=["performance_MIDI_external"])

        # Filter by annotations
        data.reset_index(inplace=True)
        data.drop(TO_IGNORE_INDICES, inplace=True)

        if opts:
            data = data[~data["performance_audio_external"].isna()]

        test_idx = data["piece_id"].isin(TEST_PIECE_IDS)

        test_data = data[test_idx]
        val_data = data[(data["piece_id"] % 10 == 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))]
        
        # We will do some processing to get some train data
        train_data = data[(data["piece_id"] % 10 != 0) & (~data["piece_id"].isin(TEST_PIECE_IDS))]

        # Before returning, we replace {ASAP} with the actual path
        actual_path = str(Path(ASAP_PATH) / "asap-dataset")
        for df in [train_data, val_data, test_data]:
            for col in ["performance_audio_external", "performance_MIDI_external", "MIDI_score_external", \
                        "performance_annotation_external", "score_annotation_external"]:
                # For all the rows in the column, replace {ASAP} with actual_path
                # But some entries may be NaN, so we need to check for that
                df.loc[:, col] = df[col].apply(lambda x: x.replace("{ASAP}", actual_path) if isinstance(x, str) else x)
        
        train_data_final = []
        val_data_final = []
        test_data_final = []

        # We want to store (performance_audio_external, performance_MIDI_external,
        # MDI_score_external, xml_score) tuples
        # Do it for train, val, test
        for _, row in train_data.iterrows():
            xml_score = str(Path(row["MIDI_score_external"]).parent / "xml_score.musicxml")
            train_data_final.append((
                row["performance_audio_external"],
                row["performance_MIDI_external"],
                row["MIDI_score_external"],
                xml_score
            ))
        
        for _, row in val_data.iterrows():
            xml_score = str(Path(row["MIDI_score_external"]).parent / "xml_score.musicxml")
            val_data_final.append((
                row["performance_audio_external"],
                row["performance_MIDI_external"],
                row["MIDI_score_external"],
                xml_score
            ))
        
        for _, row in test_data.iterrows():
            xml_score = str(Path(row["MIDI_score_external"]).parent / "xml_score.musicxml")
            test_data_final.append((
                row["performance_audio_external"],
                row["performance_MIDI_external"],
                row["MIDI_score_external"],
                xml_score
            ))
        
        # Validate the existence of all files
        for dataset, name in zip([train_data_final, val_data_final, test_data_final], ['train', 'val', 'test']):
            for audio_file, midi_file, midi_score_file, xml_score_file in dataset:
                assert Path(audio_file).exists(), f"{audio_file} does not exist in {name} set"
                assert Path(midi_file).exists(), f"{midi_file} does not exist in {name} set"
                assert Path(midi_score_file).exists(), f"{midi_score_file} does not exist in {name} set"
                assert Path(xml_score_file).exists(), f"{xml_score_file} does not exist in {name} set"
        return train_data_final, val_data_final, test_data_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract segments for user study")
    
    # Define arguments
    parser.add_argument("-nm", "--num_measures", help="Number of measures to extract")
    parser.add_argument("-p", "--path", help="Path to ASAP dataset", required=False)
    parser.add_argument("-o", "--output_dir", help="Path to directory to store segements. Default is 'data'",\
        required=False)
    parser.add_argument("-ts", "--test_split", help="Default is `maestro`. You can use `beyer` as well.", required=False)

    args = parser.parse_args()

    if args.test_split is None:
        test_split = "maestro"
    else:
        test_split = args.test_split

    if args.path:
        ASAPDataset(asap_path=args.path, num_measures=int(args.num_measures), store_dir=args.output_dir, \
                test_split=test_split)
    else:
        ASAPDataset(num_measures=int(args.num_measures), store_dir=args.output_dir, test_split=test_split)

