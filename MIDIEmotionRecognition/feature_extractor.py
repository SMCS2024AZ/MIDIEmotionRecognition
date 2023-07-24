"""Implements classes to extract static and sequential features from MIDI files.
"""

from music21 import converter, instrument, note, roman, common
import numpy as np


MAX_CHORD_SIZE = 8


def normalize_pitch(pitch):
    """Normalize pitch by dividing it by the maximum MIDI value of 127.

    Args:
        pitch (int): MIDI pitch value.

    Returns:
        float: Normalized pitch value between 0 and 1.
    """
    return pitch / 127


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
        """Represents note as a three-dimensional numpy array.  Pitch is normalized.

        Returns:
            numpy.array: Numpy array containing normalized pitch, step, and duration.
        """
        return np.array([normalize_pitch(self.pitch), self.step, self.duration], dtype=object)


class ChordFeatures:
    """Represents a single chord in a sequence.
    
    Attributes:
        pitches (list): List of pitches the chord is comprised of.
        function (str): Roman numeral representing function of chord within key.
        accidental (music21.pitch.Accidental): Accidental object representing pitch
        deviation of chord.
        is_major (bool): Whether the chord is major or not.
        Inversion (int): Number representing chord inversion.
    """
    def __init__(self, pitches, function, accidental, quality, inversion):
        self.pitches = [pitch.midi for pitch in pitches]
        self.pitches.extend([0] * (MAX_CHORD_SIZE - 1))
        self.pitches = self.pitches[0:8]
        self.function = function
        self.accidental = accidental
        self.is_major = quality == "major"
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
        """Represents chord as an 12-dimensional numpy array.  Pitches are normalized.

        Returns:
            numpy.array: Numpy array containing normalized chord pitches, roman numeral function,
            accidental, quality (major or not), and inversion
        """
        vector = []
        for pitch in self.pitches:
            vector.append(normalize_pitch(pitch))
        vector.extend([common.fromRoman(self.function),
                       self.accidental_to_int(),
                       int(self.is_major),
                       self.inversion])
        return np.array(vector, dtype=object)


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


    def get_first_note_offset(self):
        """Gets the offset of the first note in the stream.

        Returns:
            float: Offset, or time distance from beginning of song, of the first note in the stream.
        """
        for event in self.stream:
            if isinstance(event, note.Note):
                return self.get_event_offset(event)
        return None


    def get_melody_seq(self):
        """Represents the MIDI's melody as a list of vectors.

        Returns:
            list: List of vectors representing the MIDI clip's melody.
        """
        notes = []
        prev_start = self.get_first_note_offset()

        for event in self.stream.notes:
            if isinstance(event, note.Note):
                start = self.get_event_offset(event)
                temp = NoteFeatures(event.pitch.midi,
                                    start - prev_start,
                                    float(event.duration.quarterLength)).vectorize()
                if len(temp) != 3:
                    self.console.log(f"[bold red]ERROR: Invalid length of note vector ({len(temp)})")
                    self.logger.error("Invalid length of note vector")
                else:
                    notes.append(temp)
                    prev_start = start

        return notes


    def get_harmony_seq(self):
        """Represents the MIDI's harmony as a list of vectors.

        Returns:
            list: List of vectors representing the MIDI clip's harmony.
        """
        harmony = []
        chords = self.midi.chordify()

        for chord in chords.recurse().getElementsByClass("Chord"):
            numeral = roman.romanNumeralFromChord(chord, self.get_key())
            temp = ChordFeatures(chord.pitches,
                                 numeral.romanNumeralAlone,
                                 numeral.frontAlterationAccidental,
                                 numeral.quality,
                                 numeral.inversion()).vectorize()
            if len(temp) != MAX_CHORD_SIZE + 4:
                self.console.log(f"[bold red]ERROR: Invalid length of chord vector ({len(temp)})")
                self.logger.error("Invalid length of chord vector")
            else:
                harmony.append(temp)
        return harmony
