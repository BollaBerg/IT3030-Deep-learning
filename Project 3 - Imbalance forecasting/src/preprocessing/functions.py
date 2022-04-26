import pandas as pd

def clamp_column(data: pd.DataFrame,
                 column: str,
                 lower: float = -1000,
                 upper: float = 1370) -> pd.Series:
    return data[column].clip(lower=lower, upper=upper)

def split_data(data: pd.DataFrame,
               target_columns: list = ["y"]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = data.drop(columns=target_columns)
    targets = data[target_columns]

    return output, targets
