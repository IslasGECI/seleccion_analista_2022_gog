from pollos_petrel import (
    linear_model,
    predict_target_linear_model,
    read_training_dataset,
    train_linear_model,
)
import pandas as pd
from pytest import approx


def test_linear_model():
    x = 2
    slope = 1
    y_intercept = 3
    obtained_result = linear_model(x, slope, y_intercept)
    expected_result = 5
    assert obtained_result == expected_result


def test_train_linear_model():
    data = {"Masa": [1, 2, 3], "target": [2, 4, 6]}
    dataset = pd.DataFrame(data=data)
    obtained_parameters = train_linear_model(dataset)
    obtained_slope = round(obtained_parameters[0])
    expected_slope = 2.0
    assert obtained_slope == expected_slope
    obtained_y_intercept = round(obtained_parameters[1])
    expected_y_intercept = 0.0
    assert obtained_y_intercept == expected_y_intercept


def test_predict_target_linear_model():
    train_dataset = read_training_dataset()
    submission_predict_age_pollos_petrel = predict_target_linear_model(train_dataset)
    obtained_target = submission_predict_age_pollos_petrel[0]
    expected_target = 47.512
    assert expected_target == approx(obtained_target, 0.01)
