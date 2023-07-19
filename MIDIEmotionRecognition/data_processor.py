"""Implement class that can read CSV files to obtain MIDI filenames and labels,
process MIDI files with feature_extractor class, and obtain both sequential and
static data as numpy vectors/matrices that can be fed into model.
"""
import os
import pandas as pd

class DataProcessor:
    def __init__(self, data):
        self.dataset = data
        self.split = os.path.join(self.dataset, "split")
        self.train = pd.read_csv(os.path.join(self.split, "train_SL.csv"))
        self.test = pd.read_csv(os.path.join(self.split, "test_SL.csv"))
        self.val = pd.read_csv(os.path.join(self.split, "val_SL.csv"))
        self.labels = pd.read_csv(os.path.join(self.dataset, "label.csv"))
        self.midis = os.listdir(os.path.join(self.dataset, "midis"))


    def get_songs_and_labels(self, dataset):
        return dataset['songID'].tolist(), dataset['DominantQ'].tolist()


    def get_song_midis(self, song_label, song):
        prefix = f'Q{song_label}_{song}'
        return [midi for midi in self.midis if midi.startswith(prefix)]


    def prepare_dataset(self, dataset):
        self.labels.set_index("ID", inplace = True)
        songs, song_labels = self.get_songs_and_labels(dataset)
        sequences, sequence_labels = []


if __name__ == "__main__":
    print("test")
    #processor = DataProcessor("../EMOPIA_1.0")
