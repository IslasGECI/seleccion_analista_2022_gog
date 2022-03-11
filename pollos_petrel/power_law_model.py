import numpy as np
from scipy.optimize import curve_fit
from sklearn.experimental import enable_iterative_imputer  # noqa
from pollos_petrel import read_testing_dataset
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from .dummy_model import add_id


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


# Estima valores nan en datos test
def imputes_test_data() -> pd.DataFrame:
    test_dataset = read_testing_dataset()
    predicted_variables_list = list(test_dataset.columns[1:])
    test_dataset_nonid = test_dataset[predicted_variables_list]
    imputer = IterativeImputer(
        estimator=KNeighborsRegressor(),
        random_state=1,
        imputation_order="ascending",
        n_nearest_features=3,
        sample_posterior=False,
    )
    imputer.fit(test_dataset_nonid)
    impute_values = imputer.transform(test_dataset_nonid)
    test_dataset_impute = pd.DataFrame(impute_values, columns=predicted_variables_list)
    test_dataset_impute_id = add_id(test_dataset_impute, test_dataset)
    return test_dataset_impute_id
