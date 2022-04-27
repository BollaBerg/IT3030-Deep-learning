import pandas as pd

from src.preprocessing.functions import clamp_column, clamp_series

def test_clamp_y_limits_values():
    data = pd.Series([-5, -3, -1, 1, 3, 5])

    clamped = clamp_series(data, lower=-2, upper=3)
    
    assert clamped.equals(pd.Series([-2, -2, -1, 1, 3, 3]))

