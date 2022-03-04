from pollos_petrel import linear_model


def test_linear_model():
    x = 2
    slope = 1
    y_intercept = 3
    obtained_result = linear_model(x, slope, y_intercept)
    expected_result = 5
    assert obtained_result == expected_result
