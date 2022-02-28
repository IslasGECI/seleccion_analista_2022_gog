import matplotlib.pyplot as plt
import seaborn as sns
from .dummy_model import read_training_dataset


def scatter_plot(xlabel, ylabel, hue):
    fig, ax = plt.subplots()
    sns.scatterplot(data=read_training_dataset(), x=xlabel, y=ylabel, hue=hue)
<<<<<<< HEAD
    return fig, ax
=======
    return fig, ax
>>>>>>> 1680258127c3e9c18cc6d5df78381dcebce990c1
