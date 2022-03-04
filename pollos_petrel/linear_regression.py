from scipy.optimize import curve_fit
import pandas as pd


# Modelo lineal
def linear_model(x, slope, y_intercept):
    return slope * x + y_intercept


# Entrenar modelo lineal
def train_linear_model(dataset: pd.DataFrame):
    clean_dataset = dataset[~dataset.Masa.isnull()]
    parameters = curve_fit(linear_model, clean_dataset.Masa, clean_dataset.target)[0]
    return parameters
