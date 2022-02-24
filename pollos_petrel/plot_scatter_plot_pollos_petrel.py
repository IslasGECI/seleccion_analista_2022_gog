import pandas as pd
import seaborn as sns

training_dataset = pd.read_csv("train.csv")

sns.scatterplot(
    data=training_dataset, x="Longitud_tarso", y="Longitud_ala", hue="target"
)
