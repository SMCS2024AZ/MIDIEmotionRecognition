"""TODO: Implement class that can build, train, and test a model.
Run this in Google Colab to leverage Google's GPUs.
"""
import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from rich.console import Console
#from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import LSTM
#from keras.layers import Dropout
from keras.layers import Dense
from keras import optimizers


def load_data(fname, orig_shape):
    """Load melody/harmony data from txt file as numpy array.

    Args:
        fname (string): Name of data text file.
        orig_shape (tuple): Tuple describing the data's original shape.

    Returns:
        numpy.array: 3d numpy array containing melody/harmony data.
    """
    data = np.loadtxt(os.path.join("MIDIEmotionRecognition", "data", fname))
    return data.reshape(orig_shape)


def load_labels(fname):
    """Load label data from txt file as numpy array.

    Args:
        fname (string): Name of label data text file.

    Returns:
        numpy.array: 1d numpy array containing label data.
    """
    res = []
    for label in np.loadtxt(os.path.join("MIDIEmotionRecognition", "data", fname)):
        res.append(label - 1)
    return np.array(res)


def calculate_offsets(steps):
    """Calculate list of offsets based on list of steps.

    Args:
        steps (list): List of values representing the time distances between musical event onsets.

    Returns:
        list: List of offsets, or time distances from the beginning of the clip.
    """
    buffer = []
    offset = 0
    for step in steps:
        offset += step
        buffer.append(offset)
    return buffer


def combine_sequences(melody, harmony):
    """Combine melody and harmony sequences.

    Args:
        melody (numpy.array): Numpy array representing melody sequence.
        harmony (numpy.array): Numpy array representing harmony sequence.

    Returns:
        pandas.DataFrame: Dataframe representing combined melody and harmony sequences.
    """
    melody_df = pd.DataFrame(melody, columns=["note_pitch", "note_step", "note_duration"])
    melody_df["offsets"] = calculate_offsets(melody_df["note_step"])
    harmony_df = pd.DataFrame(harmony, columns=["chord_pitch0", "chord_pitch1", "chord_pitch2",
                                                "chord_pitch3", "chord_pitch4", "chord_pitch5",
                                                "chord_pitch6", "chord_pitch7", "chord_step",
                                                "chord_duration", "chord_function",
                                                "chord_function_accidental", "chord_quality",
                                                "chord_inversion"])
    harmony_df["offsets"] = calculate_offsets(harmony_df["chord_step"])
    combined = pd.merge(melody_df, harmony_df, how="outer", on="offsets")
    combined = combined.sort_values("offsets").drop_duplicates()
    return combined.ffill().drop("offsets", axis=1)


def combine_all(melody_seqs, harmony_seqs):
    """Combine list of melody sequences and harmony sequences.

    Args:
        melody_seqs (numpy.array): List of melody sequences.
        harmony_seqs (numpy.array): List of harmony sequences.

    Returns:
        list: List of combined melody and harmony sequences.
    """
    res = []
    console = Console(record=True)
    with console.status("Combining sequences", spinner="line"):
        for i, melody_seq in enumerate(melody_seqs):
            harmony_seq = harmony_seqs[i]
            res.append(combine_sequences(melody_seq, harmony_seq))
            console.log(f"[bold green]Sequence set {i + 1}/{len(melody_seqs)} completed")
    console.print("[bold green]Dataset prepared successfully!")
    return res


def len_longest_dataframe(dataframes):
    """Get the length of the longest dataframe in a list of dataframes.

    Args:
        dataframes (list): List of dataframes.

    Returns:
        int: Length of longest dataframe in given lists.
    """
    lengths = [df.shape[0] for df in dataframes]
    return max(lengths)


def pad_sequences(sequences, max_len):
    """Pad list of sequences with zeroes to a maximum length.

    Args:
        sequences (list): List of sequences.
        max_len (int): Max length of sequences.

    Returns:
        numpy.array: Numpy array of padded sequences.
    """
    res = []
    for seq in sequences:
        padded = seq.reindex(range(max_len), fill_value=0)
        res.append(padded.to_numpy())
    return np.array(res)


if __name__ == "__main__":
    train_melody = load_data("train_melody_seqs.txt", (791, 857, 3))
    train_harmony = load_data("train_harmony_seqs.txt", (791, 413, 14))
    train_features = pad_sequences(combine_all(train_melody, train_harmony), 1172)
    train_labels = load_labels("train_labels.txt")

    test_melody = load_data("test_melody_seqs.txt", (80, 857, 3))
    test_harmony = load_data("test_harmony_seqs.txt", (80, 413, 14))
    test_features = pad_sequences(combine_all(test_melody, test_harmony), 1172)
    test_labels = load_labels("test_labels.txt")

    model = Sequential()
    model.add(InputLayer(input_shape=(1172, 17)))
    model.add(LSTM(128, activation="tanh", recurrent_dropout=0.5))
    model.add(Dense(4, activation="softmax"))
    opt = optimizers.Adam()
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_features, train_labels, epochs=50, batch_size=32)
    scores = model.evaluate(test_features, test_labels, verbose=0)
    print(f"Accuracy: {(scores[1]*100)}")
