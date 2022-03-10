from pollos_petrel import (
    add_mean_as_target,
    add_id,
    drop_all_but_id,
    get_target_mean,
    read_testing_dataset,
    read_training_dataset,
    write_submission,
)
import os
import pandas as pd


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


# Agrega columna target con el promedio
def test_add_mean_as_target():
    submission_with_mean_as_target = add_mean_as_target()
    obtained_target = submission_with_mean_as_target["target"][1]
    expected_target = 34.67101226993865
    assert expected_target == obtained_target


# Guarda el archivo con sufijo _submission.csv
def test_write_submission():
    submission_path = "pollos_petrel/example_python_submission.csv"
    if os.path.exists(submission_path):
        os.remove(submission_path)
    write_submission()
    assert os.path.exists(submission_path)
    os.remove(submission_path)
