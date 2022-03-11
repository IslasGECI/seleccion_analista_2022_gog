from pollos_petrel import (
    linear_model,
    predict_age_pollos_petrel,
    train_linear_model,
    write_submission_age_pollos_petrel,
)
import pandas as pd
import os
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


def test_predict_age_pollos_petrel():
    submission_predict_age_pollos_petrel = predict_age_pollos_petrel()
    obtained_target = submission_predict_age_pollos_petrel.target[0]
    expected_target = 47.512
    assert expected_target == approx(obtained_target, 0.01)


# Guarda el archivo con sufijo _submission.csv
def test_write_submission_age_pollos_petrel():
    submission_path = "pollos_petrel/memo_1_submission.csv"
    if os.path.exists(submission_path):
        os.remove(submission_path)
    write_submission_age_pollos_petrel()
    assert os.path.exists(submission_path)
    os.remove(submission_path)
