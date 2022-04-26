import pandas as pd
import pytest

from src.preprocessing.normalizer import Normalizer

def test_normalizer_fit_transform_transforms_correctly():
    df = pd.DataFrame({
        "col_1": [0, 1, 2, 3, 4],
        "col_2": [-4, -6, 2, 9, 0],
    })
    df_copy = df.copy()

    normalizer = Normalizer()
    df["new_col_1"] = normalizer.fit_transform(df["col_1"])

    # normalizer follows formula
    assert df["new_col_1"].equals(
        (df["col_1"] - df["col_1"].min()) / (df["col_1"].max() - df["col_1"].min())
    )

    # Original column has not been changed
    assert df["col_1"].equals(df_copy["col_1"])

    # Transformed column has max = 1, min = 0
    assert df["new_col_1"].max() == 1
    assert df["new_col_1"].min() == 0


def test_normalizer_fit_without_changing_data():
    df = pd.DataFrame({
        "col_1": [0, 1, 2, 3, 4],
        "col_2": [-4, -6, 2, 9, 0],
    })
    df_copy = df.copy()

    normalizer = Normalizer()
    normalizer.fit(df["col_1"])

    assert normalizer.max == df["col_1"].max()
    assert normalizer.min == df["col_1"].min()

    assert df["col_1"].equals(df_copy["col_1"])


def test_normalizer_raises_if_not_fitted():
    df = pd.DataFrame({
        "col_1": [0, 1, 2, 3, 4],
        "col_2": [-4, -6, 2, 9, 0],
    })

    normalizer = Normalizer()
    with pytest.raises(ValueError):
        df["new_col_1"] = normalizer.transform(df["col_1"])

