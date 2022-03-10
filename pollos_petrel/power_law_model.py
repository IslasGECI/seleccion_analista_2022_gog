import numpy as np
from scipy.optimize import curve_fit


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
