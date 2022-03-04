from scipy.optimize import curve_fit
import pandas as pd
from .dummy_model import read_training_dataset, read_testing_dataset, drop_all_but_id


# Modelo lineal
def linear_model(x, slope, y_intercept):
    return slope * x + y_intercept


# Entrenar modelo lineal
def train_linear_model(dataset: pd.DataFrame):
    clean_dataset = dataset[~dataset.Masa.isnull()]
    parameters = curve_fit(linear_model, clean_dataset.Masa, clean_dataset.target)[0]
    return parameters


# Predice la edad a partir de la masa con modelo lineal
def predict_age_pollos_petrel() -> pd.DataFrame:
    train_dataset = read_training_dataset()
    test_dataset = read_testing_dataset()
    parameters = train_linear_model(train_dataset)
    predicted_target = linear_model(test_dataset.Masa, *parameters)
    submission = drop_all_but_id(test_dataset)
    submission["target"] = predicted_target
    return submission
