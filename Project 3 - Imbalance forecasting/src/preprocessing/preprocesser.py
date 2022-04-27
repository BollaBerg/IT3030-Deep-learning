import pandas as pd

from src.helpers.path import DATA_PATH
from src.preprocessing.functions import clamp_column
from src.preprocessing.standardizer import Standardizer


class NotFittedError(Exception):
    pass


class Preprocesser:
    """
    Organizer of all required preprocessing for the given datasets and tasks.

    Typical usage:
        >>> training_data = pd.DataFrame(...)
        >>> validation_data = pd.DataFrame(...)
        >>> preprocesser = Preprocesser()
        >>> training_data = preprocesser.fit_transform(training_data)
        >>> validation_data = preprocesser.transform(validation_data)
    
    """
    keep_columns = ["start_time"]
    transform_columns = ["hydro", "micro", "thermal", "wind", "total", "sys_reg", "flow"]
    if_fitted = False

    def __init__(self,
                 min_y_value: float = -1000,
                 max_y_value: float = 1370,):
        self.min_y_value = min_y_value
        self.max_y_value = max_y_value

        self.transformers = dict()
    

    def fit(self, data: pd.DataFrame):
        """Fit the Preprocesser to a DataFrame.

        Args:
            data (pd.DataFrame): Data which the Preprocesser should fit to.
        """
        for column in self.transform_columns:
            self.transformers[column] = Standardizer()
            self.transformers[column].fit(data[column])
        
        self.is_fitted = True
    

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame using the fitted Preprocesser.

        Note: Requires that the Preprocesser has previously been fitted.

        Args:
            data (pd.DataFrame): Data to transform

        Raises:
            NotFittedError: Raised if Preprocesser has not been fitted

        Returns:
            pd.DataFrame: Transformed data
        """
        if not self.is_fitted:
            raise NotFittedError("Preprocesser has not been fitted yet.")

        output = data[self.keep_columns].copy()

        for column in self.transform_columns:
            output[column] = self.transformers[column].transform(data[column])
        
        output["y"] = clamp_column(
            data, column="y", lower=self.min_y_value, upper=self.max_y_value
        )

        # TODO: Create new features
        
        return output
    

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the Preprocesser to a DataFrame, then transform the DataFrame

        Args:
            data (pd.DataFrame): Data the Preprocesser should fit to, then
                transform

        Returns:
            pd.DataFrame: Transformed DataFrame
        """
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
