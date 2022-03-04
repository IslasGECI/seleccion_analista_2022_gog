from pollos_petrel import linear_model, train_linear_model
import pandas as pd


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
    obtained_a_parameter = round(obtained_parameters[0], 1)
    expected_a_parameter = 2.0
    assert obtained_a_parameter == expected_a_parameter
    obtained_b_parameter = round(obtained_parameters[1], 1)
    expected_b_parameter = 0.0
    assert obtained_b_parameter == expected_b_parameter
