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
    """Get the length of the longest list in a list of lists.

    Args:
        lists (list): List of lists.

    Returns:
        int: Length of longest list in given lists.
    """
    lengths = [len(i) for i in lists]
    return max(lengths)


class DataProcessor:
    """Class that reads CSV files to obtain MIDI filenames and labels,
    processes MIDI files, and obtains sequential data separated into
    training, validation, and test sets.
    
    Attributes:
        dataset (str): Location of primary dataset.
        split (str): Location of train/test/val split csv files.
        train (pandas.DataFrame): Pandas dataframe containing songs labeled
        by arousal-valence emotion quadrant for training.
        test (pandas.DataFrame): Pandas dataframe containing songs labeled
        by arousal-valence emotion quadrant for testing.
        val (pandas.DataFrame): Pandas dataframe containing songs labeled
        by arousal-valence emotion quadrant for validation.
        labels (pandas.DataFrame): Pandas dataframe containing midis labeled by
        arousal-valence emotion quadrant.
        midi_folder (str): Folder containing midi files.
        midis (str): Midi filenames within midi_folder.
        train_seqs (tuple): Training sequences including melody, harmony, and labels.
        test_seqs (tuple): Testing sequences including melody, harmony, and labels.
        val_seqs (tuple): Validation sequences including melody, harmony, and labels.
        console (rich.console.Console): Rich text console.
        logger (logging.Logger): Logger for log messages.
    """
    def __init__(self, data):
        self.dataset = data
        self.split = os.path.join(self.dataset, "split")
        self.train = pd.read_csv(os.path.join(self.split, "train_SL.csv"))
        self.test = pd.read_csv(os.path.join(self.split, "test_SL.csv"))
        self.val = pd.read_csv(os.path.join(self.split, "val_SL.csv"))
        self.labels = pd.read_csv(os.path.join(self.dataset, "label.csv"))
        self.midi_folder = os.path.join(self.dataset, "midis")
        self.midis = os.listdir(os.path.join(self.dataset, "midis"))
        self.train_seqs = ()
        self.test_seqs = ()
        self.val_seqs = ()
        self.console = Console(record=True)
        logging.basicConfig(filename=os.path.join("MIDIEmotionRecognition",
                                                  "logs",
                                                  "data_processing.log"),
                        format="%(asctime)s %(message)s",
                        filemode="w")
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)


    def select_by_category(self, category, values):
        """Select a dataset by category.

        Args:
            category (str): Dataset category (train, test, val).
            values (list): List of return values.

        Raises:
            ValueError: Category must be train, test, or val.

        Returns:
            any: Depends on types contained in values list.
        """
        if category == "train":
            return values[0]
        elif category == "test":
            return values[1]
        elif category == "val":
            return values[2]
        else:
            raise ValueError("Category must be train, test, or val.")


    def set_seqs_by_category(self, category, new_value):
        """Set sequence to new value.

        Args:
            category (str): Dataset category (train, test, val).
            new_value (any): New value to set sequence to.

        Raises:
            ValueError: Category must be train, test, or val.
        """
        if category == "train":
            self.train_seqs = new_value
        elif category == "test":
            self.test_seqs = new_value
        elif category == "val":
            self.val_seqs = new_value
        else:
            raise ValueError("Category must be train, test, or val.")


    def get_dataset_by_category(self, category):
        """Get dataset by given category (train, test, val)

        Args:
            category (str): Dataset category (train, test, val).

        Returns:
            pandas.DataFrame: Pandas dataframe containing songs labeled by arousal-valence
            emotion quadrant.
        """
        return self.select_by_category(category,
                                    [self.train, self.test, self.val])


    def get_seqs_by_category(self, category):
        """Get dataset sequences by given category (train, test, val)

        Args:
            category (str): Dataset category (train, test, val).

        Returns:
            tuple: Dataset tuple (training sequences, testing sequences, validaiton sequences).
        """
        return self.select_by_category(category,
                                    [self.train_seqs, self.test_seqs, self.val_seqs])


    def get_songs_and_labels(self, dataset):
        """Gets list of song IDs and emotion labels.

        Args:
            dataset (pandas.DataFrame): Pandas dataframe containing songs labeled by arousal-valence
            emotion quadrant.

        Returns:
            tuple: Tuple containing list of slong IDs and list of labels.
        """
        return dataset['songID'].tolist(), dataset['DominantQ'].tolist()


    def get_song_midis(self, song, song_label):
        """Get midis of a song based on the song and its label.

        Args:
            song (str): Name of a song.
            song_label (str): Song's arousal-valence emotion label.

        Returns:
            list: List of midi filenames from given song.
        """
        prefix = f'Q{song_label}_{song}'
        return [midi for midi in self.midis if midi.startswith(prefix)]


    def get_midi_seqs(self, midis):
        """Extract melody, harmony, and labels from midis.

        Args:
            midis (list): List of midi filenames.

        Returns:
            tuple: Tuple with melody sequences, harmony sequences, and labels.
        """
        melody_seqs = []
        harmony_seqs = []
        seq_labels = []
        for midi in midis:
            sequences = FeatureExtractor(os.path.join(self.midi_folder, midi),
                                         self.console,
                                         self.logger).get_sequences()
            melody_seqs.append(sequences[0])
            harmony_seqs.append(sequences[1])
            seq_labels.append(self.labels.loc[midi[:-4]][0])
        return melody_seqs, harmony_seqs, seq_labels


    def pad_sequences(self, sequences, max_len):
        """Pad list of sequences with zeroes to a maximum length.

        Args:
            sequences (list): List of sequences.
            max_len (int): Max length of sequences.

        Returns:
            list: List of padded sequences.
        """
        res = []
        #max_len = len_longest_list(sequences)
        zero_vector = [0] * len(sequences[0][0])
        for sequence in sequences:
            buffer = sequence
            for i in range(max_len - len(sequence)):
                buffer.append(zero_vector)
            res.append(buffer)
        return res


    def prepare_dataset(self, category):
        """Get all melodies, harmonies, and labels from given dataset split category.

        Args:
            category (str): Dataset category (train, test, val).
        """
        dataset = self.get_dataset_by_category(category)

        self.labels.set_index("ID", inplace = True)
        songs, song_labels = self.get_songs_and_labels(dataset)
        melody_seqs = []
        harmony_seqs = []
        seq_labels = []
        length = len(dataset)

        with self.console.status(f"Preparing dataset \"{category}\"", spinner="line"):
            for i in range(length):
                song, song_label = songs[i], song_labels[i]
                message = f"Extracting MIDI data for {song}..."
                self.console.log(message)
                self.logger.info(message)
                song_midis = self.get_song_midis(song, song_label)
                midi_seqs = self.get_midi_seqs(song_midis)
                melody_seqs.extend(midi_seqs[0])
                harmony_seqs.extend(midi_seqs[1])
                seq_labels.extend(midi_seqs[2])
                self.console.log(f"[bold green]Done! ({i + 1}/{length})")
                self.logger.info("Success (%d/%d)", i + 1, length)
        self.console.print(f"[bold green]Dataset \"{category}\" prepared successfully!")

        self.set_seqs_by_category(category,
                                (np.array(self.pad_sequences(melody_seqs, 857)),
                                 np.array(self.pad_sequences(harmony_seqs, 413)),
                                 np.array(seq_labels)))


    def save_training_data(self, data_directory, category):
        """Save training data as csv file.

        Args:
            data_directory (str): Directory in which to save data.
            category (str): Dataset category (train, test, val).
        """
        names = [f"{category}_melody_seqs.txt",
                 f"{category}_harmony_seqs.txt",
                 f"{category}_labels.txt"]
        for i, name in enumerate(names):
            arr = self.get_seqs_by_category(category)[i]
            self.logger.info(str(arr.shape))
            arr_2d = arr.reshape(arr.shape[0], -1)
            np.savetxt(os.path.join(data_directory, name),
                       arr_2d, fmt="%s")


if __name__ == "__main__":
    processor = DataProcessor("EMOPIA_1.0")
    data_dir = os.path.join("MIDIEmotionRecognition", "data")
    #processor.prepare_dataset("train")
    #processor.save_training_data(data_dir, "train")
    #processor.prepare_dataset("test")
    #processor.save_training_data(data_dir, "test")
    processor.prepare_dataset("val")
    processor.save_training_data(data_dir, "val")
