from pollos_petrel import (
    imputes_test_data,
    Model,
    power_law_model,
    read_training_dataset,
    train_power_law_model,
)
import pandas as pd
import pytest


def test_power_law_model():
    x = 4
    constant_factor = 2
    power_law_index = 0.5
    y_intercept = 1
    obtained_result = power_law_model(x, constant_factor, power_law_index, y_intercept)
    expected_result = 5
    assert obtained_result == expected_result


def test_train_power_law_model():
    data = {"Longitud_ala": [1, 4, 16], "target": [3, 5, 9]}
    dataset = pd.DataFrame(data=data)
    obtained_parameters = train_power_law_model(dataset)
    obtained_constant_factor = round(obtained_parameters[0], 1)
    expected_constant_factor = 2.0
    assert obtained_constant_factor == expected_constant_factor
    obtained_power_law_index = round(obtained_parameters[1], 1)
    expected_power_law_index = 0.5
    assert obtained_power_law_index == expected_power_law_index
    obtained_y_intercept = round(obtained_parameters[2], 1)
    expected_y_intercept = 1.0
    assert obtained_y_intercept == expected_y_intercept


def test_imputes_test_data():
    test_dataset_impute = imputes_test_data()
    obtained_longitud_ala = test_dataset_impute.Longitud_ala[0]
    expected_longitud_ala = 147.5
    assert obtained_longitud_ala == expected_longitud_ala


EXPECTED_TARGET_FOR_MODEL = {"linear": 47.512, "power": 65.842}
PREDICT_TARGET_FROM_MODEL = {
    "linear": Model().LinearModel.predict_target,
    "power": Model().PowerModel.predict_target,
}


testdata = [
    (EXPECTED_TARGET_FOR_MODEL["linear"], PREDICT_TARGET_FROM_MODEL["linear"]),
    (EXPECTED_TARGET_FOR_MODEL["power"], PREDICT_TARGET_FROM_MODEL["power"]),
]


@pytest.mark.parametrize("expected, predict_target", testdata)
def test_predict_target_power_model(expected, predict_target):
    train_dataset = read_training_dataset()
    submission_predict_age_pollos_petrel = predict_target(train_dataset)
    obtained_target = submission_predict_age_pollos_petrel[0]
    assert expected == pytest.approx(obtained_target, 0.01)
