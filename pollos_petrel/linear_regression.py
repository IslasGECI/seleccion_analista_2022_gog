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


# Obtiene la edad usando un modelo lineal de la masa
def get_target_from_linear_regression(train_dataset, test_dataset):
    parameters = train_linear_model(train_dataset)
    return linear_model(test_dataset.Masa, *parameters)


# Genera formato para subir la edad
def get_submission(test_dataset, predicted_target):
    submission = drop_all_but_id(test_dataset)
    submission["target"] = predicted_target
    return submission


# Predice la edad a partir de la masa con modelo lineal
def predict_age_pollos_petrel() -> pd.DataFrame:
    train_dataset = read_training_dataset()
    test_dataset = read_testing_dataset()
    predicted_target = get_target_from_linear_regression(train_dataset, test_dataset)
    return get_submission(test_dataset, predicted_target)


# Guarda el archivo con sufijo _submission.csv
def write_submission_age_pollos_petrel():
    submission_path = "pollos_petrel/memo_1_submission.csv"
    submission = predict_age_pollos_petrel()
    submission.to_csv(submission_path)
