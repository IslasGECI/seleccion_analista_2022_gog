from pollos_petrel import power_law_model, Power_Law_Parameters


def test_power_law_model():
    x = 4
    parameters = Power_Law_Parameters(
        **{"constant_factor": 2, "power_law_index": 0.5, "y_intercept": 1}
    )
    obtained_result = power_law_model(x, parameters)
    expected_result = 5
    assert obtained_result == expected_result
