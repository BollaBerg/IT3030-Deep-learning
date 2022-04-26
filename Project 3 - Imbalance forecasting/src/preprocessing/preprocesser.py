import pandas as pd

from src.helpers.core import list_without_element
from src.helpers.path import DATA_PATH
from src.preprocessing.functions import clamp_column
from src.preprocessing.standardizer import Standardizer

class Preprocesser:
    keep_columns = ["start_time"]
    transform_columns = ["hydro", "micro", "thermal", "wind", "total", "sys_reg", "flow"]

    def __init__(self,
                 min_y_value: float = -1000,
                 max_y_value: float = 1370,):
        self.min_y_value = min_y_value
        self.max_y_value = max_y_value

        self.transformers = dict()
    

    def fit(self, data: pd.DataFrame):
        for column in self.transform_columns:
            self.transformers[column] = Standardizer()
            self.transformers[column].fit(data[column])
    

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        output = data[self.keep_columns].copy()

        for column in self.transform_columns:
            output[column] = self.transformers[column].transform(data[column])
        
        try:
            output["y"] = clamp_column(
                data, column="y", lower=self.min_y_value, upper=self.max_y_value
            )
        except KeyError:
            pass
        
        return output
    

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)


if __name__ == "__main__":
    preprocesser = Preprocesser()
    data = pd.read_csv(DATA_PATH / "train.csv", parse_dates=["start_time"])
    print("DATA")
    print(data.describe())

    processed = preprocesser.fit_transform(data)
    print("FIT_TRANSFORM")
    print(processed.describe())

    processed_test = preprocesser.transform(data.drop(columns=["y"]))
    print("TRANSFORM")
    print(processed_test.describe())