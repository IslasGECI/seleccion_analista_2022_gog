# import pandas as pd
import seaborn as sns
from .dummy_model import read_training_dataset


def scatter_plot(xlabel, ylabels, hue):
    return sns.scatterplot(data=read_training_dataset(), x=xlabels, y=ylabels, hue=hue)
