"""Implements class to extract and save sequential data, split into
training, validation, and test sets.
"""
import os
import logging
import pandas as pd
import numpy as np
from feature_extractor import FeatureExtractor
from rich.console import Console


def len_longest_list(lists):
    """Get the length of the longest nested list.
    """
    lengths = [len(i) for i in lists]
    return max(lengths)


class DataProcessor:
    """Class that reads CSV files to obtain MIDI filenames and labels,
    processes MIDI files with FeatureExtractor, and saves the obtained
    sequential and label data as .npy files.
    
    Attributes:
        dataset (str): Location of primary dataset.
        labels (pandas.DataFrame): Pandas dataframe containing midi filenames, labels,
        and annotators.
        midi_folder (str): Folder containing midi files.
        midis (str): Midi filenames within midi_folder.
        console (rich.console.Console): Rich text console.
        logger (logging.Logger): Logger for log messages.
    """
    def __init__(self, data):
        self.dataset = data
        self.labels = pd.read_csv(os.path.join(self.dataset, "label.csv"))
        self.midi_folder = os.path.join(self.dataset, "midis")
        self.midis = os.listdir(os.path.join(self.dataset, "midis"))
        self.console = Console(record=True)
        logging.basicConfig(filename=os.path.join("MIDIEmotionRecognition",
                                                  "logs",
                                                  "data_processing.log"),
                        format="%(asctime)s %(message)s",
                        filemode="w")
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)


    def get_midi_seqs(self, midi):
        """Extract melody and harmony from midi.

        Args:
            midis (str): Midi filename.

        Returns:
            tuple: Tuple with melody and harmony sequences.
        """
        melody = []
        harmony = []
        seqs = FeatureExtractor(os.path.join(self.midi_folder, midi),
                                     self.console,
                                     self.logger).get_seqs()
        melody = seqs[0]
        harmony = seqs[1]
        return melody, harmony


    def pad_seqs(self, seqs):
        """Pad list of sequences with zeroes to a maximum length.

        Args:
            sequences (list): List of sequences.

        Returns:
            list: List of padded sequences.
        """
        res = []
        max_len = len_longest_list(seqs)
        zero_vector = [0] * len(seqs[0][0])
        for sequence in seqs:
            buffer = sequence
            for _ in range(max_len - len(sequence)):
                buffer.append(zero_vector)
            res.append(buffer)
        return res


    def prepare_dataset(self, data_dir):
        """Extract all melodies, harmonies, and labels from dataset and save as .npy files.
        """
        melodies = []
        harmonies = []
        seq_labels = []
        length = self.labels.shape[0]

        with self.console.status("Preparing dataset", spinner="line"):
            for i, row in self.labels.iterrows():
                seqs = self.get_midi_seqs(row["ID"] + ".mid")
                melodies.append(seqs[0])
                harmonies.append(seqs[1])
                seq_labels.append(row["4Q"])
                self.console.log(f"[bold green]Done! ({i + 1}/{length})")
                self.logger.info("Success (%d/%d)", i + 1, length)
        self.console.print("[bold green]Dataset prepared successfully!")

        melodies = np.array(self.pad_seqs(melodies))
        harmonies = np.array(self.pad_seqs(harmonies))
        seq_labels = np.array(seq_labels)

        np.save(os.path.join(data_dir, "melodies"), melodies)
        np.save(os.path.join(data_dir, "harmonies"), harmonies)
        np.save(os.path.join(data_dir, "labels"), seq_labels)


if __name__ == "__main__":
    processor = DataProcessor("EMOPIA_1.0")
    data_loc = os.path.join("MIDIEmotionRecognition", "data")
    processor.prepare_dataset(data_loc)
 