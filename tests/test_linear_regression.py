from pollos_petrel import linear_model


def test_linear_model():
    obtained_result = linear_model(2, 1, 3)
    expected_result = 5
    assert obtained_result == expected_result
