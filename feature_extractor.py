"""Implements classes to extract static and sequential features from MIDI files.
"""

from music21 import converter, instrument, note, tempo
import numpy as np

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


    def normalize_pitch(self, pitch):
        """Normalize pitch by dividing it by the maximum MIDI value of 127.

        Args:
            pitch (int): MIDI pitch value.

        Returns:
            float: Normalized pitch value between 0 and 1.
        """
        return pitch / 127


    def vectorize(self):
        """Represents note as a three-dimensional numpy array.  Pitch is normalized.

        Returns:
            numpy.array: Numpy array containing normalized pitch, step, and duration.
        """
        return np.array([self.normalize_pitch(self.pitch), self.step, self.duration])


    def __str__(self):
        return f"Pitch: {self.pitch}, Step: {self.step:.2f}, Duration: {self.duration:.2f}"


class FeatureExtractor:
    """Extracts relevant features from midi file.
    
    Attributes:
        midi (music21.stream.base.Score): MIDI parsed by music21 converter.
        parts (music21.stream.base.Score): MIDI parts/instruments.
        piano (music21.stream.base.Part): MIDI piano part/instrument.
        stream (music21.stream.iterator.RecursiveIterator): Stream of music21 objects.
    """
    def __init__(self, filename):
        self.midi = converter.parse(filename)
        self.parts = instrument.partitionByInstrument(self.midi)
        self.piano = self.parts.parts[0]
        self.stream = self.piano.recurse()


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


    def get_melody_sequence(self):
        """Represents the MIDI's melody as a list of NoteFeature objects.
        TODO: Refine this to extract a more accurate melody (e.g. take top notes from chords,
        ignore notes that are pitched too low, etc.)

        Returns:
            list: List of NoteFeature objects representing the MIDI clip's melody.
        """
        notes = []
        prev_start = self.get_first_note_offset()

        for event in self.stream.notes:
            if isinstance(event, note.Note):
                start = self.get_event_offset(event)
                notes.append(NoteFeatures(event.pitch.midi,
                                          start - prev_start,
                                          float(event.duration.quarterLength)))
                prev_start = start

        return notes


    def get_key(self):
        """Gets estimated key of MIDI clip.

        Returns:
            str: Estimated key of MIDI clip.
        """
        return self.midi.analyze("key").name


    def get_tempo(self):
        """Gets tempo of MIDI.
        
        Returns:
            float: MIDI tempo in BPM.
        """
        for event in self.stream:
            if isinstance(event, tempo.MetronomeMark):
                return event.getQuarterBPM()
        return None



if __name__ == "__main__":
    extractor = FeatureExtractor("test_midi.mid")
    print(extractor.get_tempo())
    #for note in extractor.get_melody_sequence():
        #print(note)
    print(extractor.get_key())
