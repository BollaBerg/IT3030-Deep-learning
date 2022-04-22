import pandas as pd

def clamp_y(data: pd.DataFrame,
            lower: float = -1000,
            upper: float = 1370):
    data.y = data.y.clip(lower=lower, upper=upper)
