import numpy as np
from scipy.optimize import curve_fit
from pollos_petrel import read_testing_dataset
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from .dummy_model import add_id, DummyModel, get_submission, read_training_dataset
from .linear_regression import LinearModel
import pydantic


# Modelo ley de potencia
def power_law_model(x, constant_factor, power_law_index, y_intercept):
    return constant_factor * np.power(x, power_law_index) + y_intercept


# Entrenar modelo ley de potencia
def train_power_law_model(dataset):
    clean_dataset = dataset[~dataset.Longitud_ala.isnull()]
    parameters = curve_fit(
        f=power_law_model, xdata=clean_dataset.Longitud_ala, ydata=clean_dataset.target
    )[0]
    return parameters


# Remover id de set de datos test
def get_test_dataset_without_id(test_dataset):
    predicted_variables_list = list(test_dataset.columns[1:])
    return test_dataset[predicted_variables_list]


# Inicializar funcion imputer
def init_imputer():
    imputer = IterativeImputer(
        estimator=KNeighborsRegressor(),
        random_state=1,
        imputation_order="ascending",
        n_nearest_features=3,
        sample_posterior=False,
    )
    return imputer


# Aplica impute a set de datos test
def impute_test_dataset(test_dataset_nonid, imputer):
    predicted_variables_list = list(test_dataset_nonid.columns)
    imputer.fit(test_dataset_nonid)
    impute_values = imputer.transform(test_dataset_nonid)
    return pd.DataFrame(data=impute_values, columns=predicted_variables_list)


# Estima valores nan en datos test
def imputes_test_data() -> pd.DataFrame:
    test_dataset = read_testing_dataset()
    test_dataset_nonid = get_test_dataset_without_id(test_dataset)
    imputer = init_imputer()
    test_dataset_imputed = impute_test_dataset(test_dataset_nonid, imputer)
    return add_id(test_dataset_imputed, test_dataset)


# Obtiene la edad usando la ley de potencia
def get_target_from_power_law(train_dataset, imputed_test_dataset):
    parameters = train_power_law_model(train_dataset)
    return power_law_model(imputed_test_dataset.Longitud_ala, *parameters)


class PowerModel:
    submission_path = "pollos_petrel/memo_2_submission.csv"

    def predict_target(train_dataset) -> pd.DataFrame:
        imputed_test_dataset = imputes_test_data()
        return get_target_from_power_law(train_dataset, imputed_test_dataset)


class Model(pydantic.BaseModel):
    DummyModel = DummyModel
    LinearModel = LinearModel
    PowerModel = PowerModel


# Guarda el archivo con0 sufijo _submission.csv
def write_submission(Model):
    train_dataset = read_training_dataset()
    test_dataset = read_testing_dataset()
    predicted_target = Model.predict_target(train_dataset)
    submission = get_submission(test_dataset, predicted_target)
    submission.to_csv(Model.submission_path)
    return submission
