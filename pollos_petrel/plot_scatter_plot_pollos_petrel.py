import seaborn as sns
from .dummy_model import read_training_dataset


def scatter_plot(xlabel, ylabel, hue):
    return sns.scatterplot(data=read_training_dataset(), x=xlabel, y=ylabel, hue=hue)
