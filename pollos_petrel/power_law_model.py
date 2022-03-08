import numpy as np


# Modelo ley de potencia
def power_law_model(x, constant_factor, power_law_index, y_intercept):
    return constant_factor * np.power(x, power_law_index) + y_intercept
