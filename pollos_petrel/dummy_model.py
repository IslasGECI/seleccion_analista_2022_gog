import pandas as pd


# Lee train.csv
def read_training_dataset() -> pd.DataFrame:
    training_dataset_path = "pollos_petrel/train.csv"
    training_dataset = pd.read_csv(training_dataset_path)
    return training_dataset


# Calcula promedio de target
def get_target_mean(dataset: pd.DataFrame) -> float:
    mean_target = dataset["target"].mean()
    return mean_target


# Lee test.csv
def read_testing_dataset() -> pd.DataFrame:
    testing_dataset_path = "pollos_petrel/test.csv"
    testing_dataset = pd.read_csv(testing_dataset_path)
    return testing_dataset


# Tira todas las columnas excepto id
def drop_all_but_id(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset_only_id = dataset[["id"]]
    return dataset_only_id


# Agrega la columna id
def add_id(dataset_previous_id, dataset_with_id):
    id_values = dataset_with_id.id
    dataset_previous_id["id"] = id_values
    return dataset_previous_id


# Genera formato para subir la edad
def get_submission(test_dataset, predicted_target):
    submission = drop_all_but_id(test_dataset)
    submission["target"] = predicted_target
    return submission


# Agrega columna target con el promedio
def add_mean_as_target() -> pd.DataFrame:
    training_dataset = read_training_dataset()
    testing_dataset = read_testing_dataset()
    predicted_target = get_target_mean(training_dataset)
    return get_submission(testing_dataset, predicted_target)


# Guarda el archivo con sufijo _submission.csv
def write_submission():
    submission_path = "pollos_petrel/example_python_submission.csv"
    submission = add_mean_as_target()
    submission.to_csv(submission_path)
