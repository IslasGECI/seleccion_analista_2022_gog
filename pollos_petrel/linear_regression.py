from scipy.optimize import curve_fit
import pandas as pd
from .dummy_model import (
    read_training_dataset,
    read_testing_dataset,
    get_submission,
)


# Modelo lineal
def linear_model(x, slope, y_intercept):
    return slope * x + y_intercept


# Entrenar modelo lineal
def train_linear_model(dataset: pd.DataFrame):
    clean_dataset = dataset[~dataset.Masa.isnull()]
    parameters = curve_fit(linear_model, clean_dataset.Masa, clean_dataset.target)[0]
    return parameters


# Obtiene la edad usando un modelo lineal de la masa
def get_target_from_linear_regression(train_dataset, test_dataset):
    parameters = train_linear_model(train_dataset)
    return linear_model(test_dataset.Masa, *parameters)


# Predice la edad a partir de la masa con modelo lineal
def predict_age_pollos_petrel() -> pd.DataFrame:
    train_dataset = read_training_dataset()
    test_dataset = read_testing_dataset()
    predicted_target = get_target_from_linear_regression(train_dataset, test_dataset)
    return get_submission(test_dataset, predicted_target)
