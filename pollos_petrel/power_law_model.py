import numpy as np
import pydantic


class Power_Law_Parameters(pydantic.BaseModel):
    constant_factor: float
    power_law_index: float
    y_intercept: float


# Modelo ley de potencia
def power_law_model(x, parameters):
    return (
        parameters.constant_factor * np.power(x, parameters.power_law_index)
        + parameters.y_intercept
    )
