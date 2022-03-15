from pollos_petrel import add_id, Model, write_submission
import os
import pandas as pd
import pytest


def test_add_id():
    data_previous_id = {"target": [3, 4]}
    dataset_previous_id = pd.DataFrame(data=data_previous_id)
    data_with_id = {"id": [1, 2], "target": [3, 4]}
    dataset_with_id = pd.DataFrame(data=data_with_id)
    dataset_add_id = add_id(dataset_previous_id, dataset_with_id)
    obtained_id = dataset_add_id.id[0]
    expected_id = 1
    assert obtained_id == expected_id
    obtained_columns = list(dataset_add_id)
    assert "id" in obtained_columns


def remove_submission(submission_path):
    if os.path.exists(submission_path):
        os.remove(submission_path)


def compare_none_rows(submission):
    number_rows = len(submission)
    none_rows = submission.target.isnull().sum()
    assert number_rows != none_rows


def compare_path_exists(submission_path):
    assert os.path.exists(submission_path)
    os.remove(submission_path)


def compare_path_and_none_rows(submission_path, submission):
    compare_path_exists(submission_path)
    compare_none_rows(submission)


SUBMISSION_PATHS = {
    "dummy": Model.DummyModel.submission_path,
    "linear": Model.LinearModel.submission_path,
    "power": Model.PowerModel.submission_path,
}


MODEL_SELECTION = {
    "dummy": Model.DummyModel,
    "linear": Model.LinearModel,
    "power": Model.PowerModel,
}

testdata = [
    (SUBMISSION_PATHS["dummy"], MODEL_SELECTION["dummy"]),
    (SUBMISSION_PATHS["linear"], MODEL_SELECTION["linear"]),
    (SUBMISSION_PATHS["power"], MODEL_SELECTION["power"]),
]


@pytest.mark.parametrize("submission_path, model_selection", testdata)
def test_write_submission(submission_path, model_selection):
    remove_submission(submission_path)
    submission = write_submission(model_selection)
    compare_path_and_none_rows(submission_path, submission)
