import pandas as pd
import torch

def clamp_column(data: pd.DataFrame,
                 column: str,
                 lower: float = -1000,
                 upper: float = 1370) -> pd.Series:
    return data[column].clip(lower=lower, upper=upper)

def clamp_series(data: pd.Series,
                 lower: float = -1000,
                 upper: float = 1370) -> pd.Series:
    return data.clip(lower=lower, upper=upper)


def split_data(data: pd.DataFrame,
               target_columns: list = ["target"]
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = data.drop(columns=target_columns)
    targets = data[target_columns]

    return output, targets


def pd_to_tensor(data: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(data.values)
