import pandas as pd

from src.preprocessing.functions import clamp_column

def test_clamp_y_limits_values():
    df = pd.DataFrame(
        {"y": [-5, -3, -1, 1, 3, 5],
        "other_data": list(range(6))}
    )
    clamped = clamp_column(df, "y", lower=-2, upper=3)
    
    assert clamped.equals(pd.Series([-2, -2, -1, 1, 3, 3]))

    clamped_other = clamp_column(df, "other_data", lower=3, upper=4)

    assert clamped_other.equals(pd.Series([3, 3, 3, 3, 4, 4]))

    # Test that clamp_column leaves column as original
    assert df.y.equals(pd.Series([-5, -3, -1, 1, 3, 5]))
    assert df.other_data.equals(pd.Series(list(range(6))))
