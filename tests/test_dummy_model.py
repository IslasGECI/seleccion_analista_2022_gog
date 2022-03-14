from pollos_petrel import (
    add_id,
    Model,
    drop_all_but_id,
    get_target_mean,
    get_submission,
    read_testing_dataset,
    read_training_dataset,
    write_submission,
)
import os
import pandas as pd
from pytest import approx


# Lee train.csv
def test_read_training_dataset():
    training_dataset = read_training_dataset()
    obtained_n_rows = training_dataset.shape[0]
    expected_n_rows = 1304
    assert expected_n_rows == obtained_n_rows


# Calcula promedio de target
def test_get_target_mean():
    data = {"id": [1, 2], "target": [3, 4]}
    dataset = pd.DataFrame(data=data)
    obtained_mean = get_target_mean(dataset)
    expected_mean = 3.5
    assert expected_mean == obtained_mean


# Lee test.csv
def test_read_testing_dataset():
    testing_dataset = read_testing_dataset()
    obtained_n_rows = testing_dataset.shape[0]
    expected_n_rows = 326
    assert expected_n_rows == obtained_n_rows


# Tira todas las columnas excepto id
def test_drop_all_but_id():
    data = {"id": [1, 2], "target": [3, 4]}
    dataset = pd.DataFrame(data=data)
    dataset_only_id = drop_all_but_id(dataset)
    obtained_columns = list(dataset_only_id.columns)
    expected_columns = ["id"]
    assert expected_columns == obtained_columns


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


def test_get_submission():
    train_dataset = read_training_dataset()
    predicted_target = Model().DummyModel.predict_target(train_dataset)
    submission = get_submission(train_dataset, predicted_target)
    assert "target" in submission.columns
    number_rows = len(submission)
    none_rows = submission.target.isnull().sum()
    assert number_rows != none_rows


# Agrega columna target con el promedio
def test_predict_target():
    train_dataset = read_training_dataset()
    submission_with_mean_as_target = Model.DummyModel.predict_target(train_dataset)
    obtained_target = submission_with_mean_as_target
    expected_target = 34.671
    assert expected_target == approx(obtained_target, 0.01)


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


def test_write_submission():
    for model_type in dir(Model)[1:4]:
        if model_type == "DummyModel":
            submission_path = Model.DummyModel.submission_path
            remove_submission(submission_path)
            submission = write_submission(Model.DummyModel)
            compare_path_and_none_rows(submission_path, submission)
        if model_type == "LinearModel":
            submission_path = Model.LinearModel.submission_path
            remove_submission(submission_path)
            submission = write_submission(Model.LinearModel)
            compare_path_and_none_rows(submission_path, submission)
        if model_type == "PowerModel":
            submission_path = Model.PowerModel.submission_path
            remove_submission(submission_path)
            submission = write_submission(Model.PowerModel)
            compare_path_and_none_rows(submission_path, submission)
