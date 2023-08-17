"""Implements classes to extract static and sequential features from MIDI files.
"""


from music21 import converter, instrument, note, chord, roman, common


MAX_CHORD_SIZE = 8
CHORD_QUALITIES = {
    "major": 0,
    "minor": 1,
    "diminished": 2,
    "augmented": 3,
    "other": 4
}


def normalize_pitch(pitch):
    """Normalize pitch by taking the modulo of the pitch value and 12.

    Args:
        pitch (int): MIDI pitch value.

    Returns:
        float: Normalized pitch value between 0 and 11 (pitch class).
    """
    return pitch % 12


class NoteFeatures:
    """Represents a single note in a sequence.
    
    Attributes:
        pitch (int): The note's MIDI pitch.
        step (float): The time difference between the current and previous note onset.
        duration (float): The note's length in quarter notes.
    """
    def __init__(self, pitch, step, duration):
        self.pitch = pitch
        self.step = step
        self.duration = duration


    def vectorize(self):
        """Represents note as a three-dimensional array.  Pitch is normalized.

        Returns:
            list: Array containing normalized pitch, step, and duration.
        """
        return [normalize_pitch(self.pitch), self.step, self.duration]


class ChordFeatures:
    """Represents a single chord in a sequence.
    
    Attributes:
        pitches (list): List of pitches the chord is comprised of.
        step (float): The time difference between the current and previous chord onset.
        duration (float): The chord's length in quarter notes.
        function (str): Roman numeral representing function of chord within key.
        accidental (music21.pitch.Accidental): Accidental object representing pitch
        deviation of chord.
        is_major (bool): Whether the chord is major or not.
        inversion (int): Number representing chord inversion.
    """
    def __init__(self, pitches, step, duration, function, accidental, quality, inversion):
        self.pitches = [pitch.midi for pitch in pitches]
        self.pitches.extend([0] * (MAX_CHORD_SIZE - 1))
        self.pitches = self.pitches[0:8]
        self.step = step
        self.duration = duration
        self.function = function
        self.accidental = accidental
        self.quality = CHORD_QUALITIES[quality]
        self.inversion = inversion


    def accidental_to_int(self):
        """Converts accidental to integer.

        Returns:
            float: How many semitones the accidental alters by.
        """
        if self.accidental is not None:
            return self.accidental.alter
        return 0


    def vectorize(self):
        """Represents chord as an 12-dimensional array.  Pitches are normalized.

        Returns:
            list: Array containing normalized chord pitches, roman numeral function,
            accidental, quality (major or not), and inversion
        """
        vector = []
        for pitch in self.pitches:
            vector.append(normalize_pitch(pitch))
        vector.extend([self.step,
                       self.duration,
                       common.fromRoman(self.function),
                       self.accidental_to_int(),
                       self.quality,
                       self.inversion])
        return vector


class FeatureExtractor:
    """Extracts relevant features from midi file.
    
    Attributes:
        midi (music21.stream.base.Score): MIDI parsed by music21 converter.
        parts (music21.stream.base.Score): MIDI parts/instruments.
        piano (music21.stream.base.Part): MIDI piano part/instrument.
        stream (music21.stream.iterator.RecursiveIterator): Stream of music21 objects.
        console (rich.console.Console): Console to print progress/errors to.
    """
    def __init__(self, filename, console, logger):
        self.midi = converter.parse(filename)
        self.parts = instrument.partitionByInstrument(self.midi)
        self.piano = self.parts.parts[0]
        self.stream = self.piano.recurse()
        self.console = console
        self.logger = logger


    def get_key(self):
        """Gets estimated key of MIDI clip.

        Returns:
            music21.key: Estimated key of MIDI clip.
        """
        return self.midi.analyze("key")


    def get_event_offset(self, event):
        """Gets the offset of a music21 object.

        Args:
            event (music21.base.Music21Object): A music21 object.

        Returns:
            float: Offset, or time distance from beginning of song, of given music21 object.
        """
        for site in event.contextSites():
            if site[0] is self.piano:
                offset = site[1]
        return float(offset)


    def get_first_event_offset(self, event_type):
        """Gets the offset of the first event of a given type in the stream.

        Returns:
            float: Offset, or time distance from beginning of song, of the first event
            of a given type in the stream.
        """
        for event in self.stream:
            if isinstance(event, event_type):
                return self.get_event_offset(event)
        return None


    def validate_vector(self, vector, required_length, name, arr, prev_start, start):
        """Validates vector, adding it to a given array if valid, updating event onset tracking,
        and allowing for logging.

        Args:
            vector (list): Vector representing a musical event.
            required_length (int): Required length of vector.
            name (str): Name of musical event represented by vector.
            arr (list): Array to add vector to if vector is valid.
            prev_start (float): Previous musical event onset.
            start (float): Current musical event onset.

        Returns:
            int: Current or previous musical event onset, depending on if
            vector is valid or not, respectively.
        """
        if len(vector) != required_length:
            self.console.log(f"[bold red]ERROR: Invalid length of {name} vector ({len(vector)})")
            self.logger.error("Invalid length of %s", name)
            return prev_start
        else:
            arr.append(vector)
            return start


    def get_seqs(self):
        """Represents the MIDI's melody as a list of vectors.

        Returns:
            list: List of vectors representing the MIDI clip's melody.
        """
        notes = []
        harmony = []
        prev_note_start = self.get_first_event_offset(note.Note)
        prev_chord_start = self.get_first_event_offset(chord.Chord)

        for event in self.stream:
            if isinstance(event, note.Note):
                start = self.get_event_offset(event)
                temp = NoteFeatures(event.pitch.midi,
                                    start - prev_note_start,
                                    float(event.duration.quarterLength)).vectorize()
                prev_note_start = self.validate_vector(temp,
                                                       3,
                                                       "note",
                                                       notes,
                                                       prev_note_start,
                                                       start)
            elif isinstance(event, chord.Chord):
                start = self.get_event_offset(event)
                numeral = roman.romanNumeralFromChord(event, self.get_key())
                step = start - prev_chord_start
                duration = float(event.duration.quarterLength)

                if numeral.romanNumeralAlone.lower() != "sw":
                    temp_chord = ChordFeatures(event.pitches,
                                            step,
                                            duration,
                                            numeral.romanNumeralAlone,
                                            numeral.frontAlterationAccidental,
                                            numeral.quality,
                                            numeral.inversion()).vectorize()
                    temp_top_note = NoteFeatures(event.pitches[0].midi,
                                                step,
                                                duration).vectorize()
                    prev_chord_start = self.validate_vector(temp_chord,
                                                            MAX_CHORD_SIZE + 6,
                                                            "chord", harmony,
                                                            prev_chord_start,
                                                            start)
                    prev_note_start = self.validate_vector(temp_top_note,
                                                        3,
                                                        "note",
                                                        notes,
                                                        prev_note_start,
                                                        start)

        return notes, harmony
