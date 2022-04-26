import pandas as pd

from src.preprocessing import clamp_column

def test_clamp_y_limits_values():
    df = pd.DataFrame(
        {"y": [-5, -3, -1, 1, 3, 5],
        "other_data": list(range(6))}
    )
    df_clamped = df.copy()
    
    clamp_column(df_clamped, "y", lower=-2, upper=3)
    
    assert df.other_data.equals(df_clamped.other_data)
    assert df_clamped.y.equals(
        pd.Series([-2, -2, -1, 1, 3, 3])
    )

    clamp_column(df_clamped, "other_data", lower=3, upper=4)

    assert df_clamped.other_data.equals(
        pd.Series([3, 3, 3, 3, 4, 4])
    )