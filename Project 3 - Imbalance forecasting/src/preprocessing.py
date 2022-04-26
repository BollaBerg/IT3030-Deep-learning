import pandas as pd

def clamp_column(data: pd.DataFrame,
                 column: str,
                 lower: float = -1000,
                 upper: float = 1370):
    data[column] = data[column].clip(lower=lower, upper=upper)
