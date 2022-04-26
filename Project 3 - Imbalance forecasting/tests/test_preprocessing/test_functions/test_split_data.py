import pandas as pd

from src.preprocessing.functions import split_data

def test_split_data_splits_columns_correctly():
    data = pd.DataFrame({
        "input_1": [-1, 3, -2, 0, 4],
        "input_2": [0, 9, 3, 4, 1],
        "y": [-1, -2, -3, -4, -5]
    })

    X, y = split_data(data, target_columns=["y"])

    assert len(X.columns) == 2
    assert X.columns[0] == "input_1"
    assert X.columns[1] == "input_2"
    assert len(y.columns) == 1
    assert y.columns[0] == "y"


def test_split_data_leaves_values():
    data = pd.DataFrame({
        "input_1": [-1, 3, -2, 0, 4],
        "input_2": [0, 9, 3, 4, 1],
        "y": [-1, -2, -3, -4, -5]
    })

    X, y = split_data(data, target_columns=["y"])

    assert X["input_1"].equals(data["input_1"])
    assert X["input_2"].equals(data["input_2"])
    assert y["y"].equals(data["y"])
