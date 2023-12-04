import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataGrapher:
    def __init__(self, melodies, harmonies, labels):
        self.melodies = melodies
        self.harmonies = harmonies
        self.labels = labels


    def get_stat(self, group, col, func):
        res = []
        for seq in group:
            stripped = seq[~np.all(seq == 0, axis=1)]
            res.append(func(stripped, axis=0)[col])
        return res


    def get_stat_type(self, stat_type, group, col):
        if stat_type == "mean":
            return self.get_stat(group, col, np.mean)
        if stat_type =="median":
            return self.get_stat(group, col, np.median)
        if stat_type =="std":
            return self.get_stat(group, col, np.std)


    def get_melody_stat(self, stat, category):
        categories = {
            "step": 1,
            "duration": 2
        }

        return self.get_stat_type(stat, self.melodies, categories[category])


    def get_harmony_stat(self, stat, category):
        categories = {
            "step": 8,
            "duration": 9
        }

        return self.get_stat_type(stat, self.harmonies, categories[category])


    def graph_stat(self, group, stat, category):
        if group =="melody":
            y = self.get_melody_stat(stat, category)
        elif group == "harmony":
            y = self.get_harmony_stat(stat, category)

        plt.scatter(self.labels, y, alpha=0.03)
        plt.xticks([1,2,3,4])
        plt.show()


    def get_harmonies_by_label(self, label):
        indices = [i for i in range(len(labels)) if labels[i] == label]
        return self.harmonies[min(indices):max(indices) + 1]


    def format_crosstab(self, crosstab):
        for i in range(1, 8):
            if float(i) not in list(crosstab.columns):
                crosstab.insert(i - 1, i, 0)
        for j in range(5):
            if float(j) not in list(crosstab.index):
                crosstab.loc[float(j)] = [0] * 7
        return crosstab.sort_index().reset_index(drop=True)


    def graph_chord_functions_qualities(self, label):
        seqs = self.get_harmonies_by_label(label)
        counts = pd.DataFrame(0, index=range(5), columns=range(1, 8))
        for seq in seqs:
            seq_data = pd.DataFrame(seq)
            no_zeros = seq_data.loc[(seq_data != 0).any(axis=1)]
            crosstab = pd.crosstab(no_zeros[12], no_zeros[10], margins=False)
            crosstab = crosstab.rename_axis(columns=None, index=None)
            counts = (counts + self.format_crosstab(crosstab)).fillna(0)
        _, axes = plt.subplots()
        axes.matshow(counts)
        axes.set_xticklabels(range(0, 8))
        axes.set_yticklabels(["", "major", "minor", "diminished", "augmented", "other"])
        mean = counts.mean().mean()
        for (i, j), z in np.ndenumerate(counts):
            axes.text(j, i, str(int(z)), ha="center", va="center",
                      color=("black" if z > mean else "white"))
        plt.show()


if __name__ == "__main__":
    data_folder = os.path.join("MIDIEmotionRecognition", "data")
    melodies = np.load(os.path.join(data_folder, "melodies.npy"))
    harmonies = np.load(os.path.join(data_folder, "harmonies.npy"))
    labels = np.load(os.path.join(data_folder, "labels.npy"))

    grapher = DataGrapher(melodies, harmonies, labels)
    grapher.graph_chord_functions_qualities(4)
