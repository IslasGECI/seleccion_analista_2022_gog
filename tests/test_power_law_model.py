from pollos_petrel import (
    imputes_test_data,
    power_law_model,
    predict_age_pollos_petrel_power_law,
    train_power_law_model,
)
import pandas as pd


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


def test_predict_age_pollos_petrel_power_law():
    submission_predict_age_pollos_petrel = predict_age_pollos_petrel_power_law()
    obtained_target = submission_predict_age_pollos_petrel.target[0]
    expected_target = 65.84275146861765
    assert expected_target == obtained_target
