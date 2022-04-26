import pandas as pd
import pytest

from src.preprocessing.standardizer import Standardizer

def test_standardizer_fit_transform_transforms_correctly():
    df = pd.DataFrame({
        "col_1": [0, 1, 2, 3, 4],
        "col_2": [-4, -6, 2, 9, 0],
    })
    df_copy = df.copy()

    standardizer = Standardizer()
    df["new_col_1"] = standardizer.fit_transform(df["col_1"])

    # standardizer follows formula
    assert df["new_col_1"].equals(
        (df["col_1"] - df["col_1"].mean()) / df["col_1"].std()
    )

    # Original column has not been changed
    assert df["col_1"].equals(df_copy["col_1"])

    # Transformed column has std = 1, mean = 0
    assert df["new_col_1"].std() == pytest.approx(1)
    assert df["new_col_1"].mean() == pytest.approx(0)


def test_standardizer_fit_without_changing_data():
    df = pd.DataFrame({
        "col_1": [0, 1, 2, 3, 4],
        "col_2": [-4, -6, 2, 9, 0],
    })
    df_copy = df.copy()

    standardizer = Standardizer()
    standardizer.fit(df["col_1"])

    assert standardizer.std == df["col_1"].std()
    assert standardizer.mean == df["col_1"].mean()

    assert df["col_1"].equals(df_copy["col_1"])


def test_standardizer_raises_if_not_fitted():
    df = pd.DataFrame({
        "col_1": [0, 1, 2, 3, 4],
        "col_2": [-4, -6, 2, 9, 0],
    })

    standardizer = Standardizer()
    with pytest.raises(ValueError):
        df["new_col_1"] = standardizer.transform(df["col_1"])

