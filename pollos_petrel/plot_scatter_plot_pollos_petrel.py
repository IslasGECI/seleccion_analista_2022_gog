import matplotlib.pyplot as plt
import seaborn as sns
from .dummy_model import read_training_dataset


def scatter_plot(xlabel, ylabel, hue):
    fig, ax = plt.subplots()
    sns.scatterplot(data=read_training_dataset(), x=xlabel, y=ylabel, hue=hue)
    return fig, ax


def plot_tarsus_vs_wing():
    fig, ax = scatter_plot("Longitud_ala", "Longitud_tarso", "target")
    return fig, ax
