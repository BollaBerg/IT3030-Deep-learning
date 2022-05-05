import numpy as np
from numpy import sin, cos, pi
import pandas as pd

from src.helpers.path import DATA_PATH
from src.preprocessing.functions import clamp_series
from src.preprocessing.normalizer import Normalizer
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
    keep_columns = []
    transform_columns = ["hydro", "micro", "thermal", "wind", "total", "sys_reg", "flow"]
    is_fitted = False

    def __init__(self,
                 min_y_value: float = -1000,
                 max_y_value: float = 1370,
                 time_of_day: bool = False,
                 time_of_week: bool = False,
                 time_of_year: bool = False,
                 last_day_y: bool = False,
                 two_last_day_y: bool = False,
                 randomize_last_y: bool = False,
                 task_2_window: int = None):
        """Create an instance of Preprocesser

        Args:
            min_y_value (float, optional): Minimum value of all target-related
                columns, for clamping. Defaults to -1000.
            max_y_value (float, optional): Maximum value of all target-related
                columns, for clamping. Defaults to 1370.
            time_of_day (bool, optional): Whether the data should include a
                representation of time of day. Defaults to False.
            time_of_week (bool, optional): Whether the data should include a
                representation of time of week / weekday. Defaults to False.
            time_of_year (bool, optional): Whether the data should include a
                representation of time of year / yearday. Defaults to False.
            last_day_y (bool, optional): Whether the data should include target
                from yesterday as an input. Defaults to False.
            two_last_day_y (bool, optional): Whether the data should include
                target from two days ago as an input. Defaults to False.
            randomize_last_y (bool, optional): Whether the last_y column should
                include randomness, to limit its importance in the predictions
                (possibly leading to better long-term predictions). Defaults
                to False.
            task_2_window (int, optional): Window to use for task 2. If None,
                no window will be applied, and it is assumed that the model is
                to solve task 1 (i.e. raw targets, not structural difference,
                will be used as targets). Defaults to None.
        """
        self.min_y_value = min_y_value
        self.max_y_value = max_y_value

        self.time_of_day = time_of_day
        self.time_of_week = time_of_week
        self.time_of_year = time_of_year
        self.last_day_y = last_day_y
        self.two_last_day_y = two_last_day_y
        self.randomize_last_y = randomize_last_y

        self.task_2_window = task_2_window

        self.transformers = dict()
    

    def fit(self, data: pd.DataFrame):
        """Fit the Preprocesser to a DataFrame.

        Args:
            data (pd.DataFrame): Data which the Preprocesser should fit to.
        """
        for column in self.transform_columns:
            self.transformers[column] = Standardizer()
            self.transformers[column].fit(data[column])
        
        self.y_transformer = Normalizer()

        clamped_y = clamp_series(data["y"], self.min_y_value, self.max_y_value)


        ############### TRANSFORM TARGET FOR TASK 2 ###############
        # For more info, see comment in self.transform
        if self.task_2_window is not None:
            # Use interpolation of average of 6 hours
            _window = self.task_2_window
            # Calculate the mean over this window, centered around each element
            averages = clamped_y.rolling(_window, center=True).mean()
            # Use numpy to create an interpolation for each value
            interpolation = np.interp(clamped_y.index, averages.index, averages)

            # .rolling() leaves us with _window - 1 NaNs, rounded up at head,
            # down at tail. We fill these with the mean of the "remaining" 
            # values
            interpolation[:int(_window/2)] = np.mean(
                clamped_y[:int(_window/2)]
            )
            interpolation[-(int(_window/2) - 1):] = np.mean(
                clamped_y[-(int(_window/2) - 1):]
            )

            # Finally, replace target column with target - interpolation
            clamped_y = clamped_y - interpolation
        ###############   END OF TASK 2 SPECIFICS   ###############


        self.y_transformer.fit(clamped_y)
        
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
        

        clamped_y = clamp_series(
            data["y"], lower=self.min_y_value, upper=self.max_y_value
        )


        ############### TRANSFORM TARGET FOR TASK 2 ###############
        # Task 2 (or 5.2.2) wants the model to target the structural difference
        # between the continuously, slowly changing consumption in the system
        # vs the abruptly changing power production of regulated power plants
        # For more info about the method (and a plot of the result), see
        #   exploration / data_exploration.ipynb
        if self.task_2_window is not None:
            # Use interpolation of average of 6 hours
            _window = self.task_2_window
            # Calculate the mean over this window, centered around each element
            averages = clamped_y.rolling(_window, center=True).mean()
            # Use numpy to create an interpolation for each value
            interpolation = np.interp(clamped_y.index, averages.index, averages)

            # .rolling() leaves us with _window - 1 NaNs, rounded up at head,
            # down at tail. We fill these with the mean of the "remaining" 
            # values
            interpolation[:int(_window/2)] = np.mean(
                clamped_y[:int(_window/2)]
            )
            interpolation[-(int(_window/2) - 1):] = np.mean(
                clamped_y[-(int(_window/2) - 1):]
            )

            # Finally, replace target column with target - interpolation
            clamped_y = clamped_y - interpolation
        ###############   END OF TASK 2 SPECIFICS   ###############


        output["target"] = self.y_transformer.transform(clamped_y)
        output["last_y"] = output["target"].shift(1)

        # Create new features
        time_columns = []

        # Create time-of-day feature
        ## We use a two-component representation if the time, using sin and cos
        ## This has two useful properties:
        ##  1) Always between 0 and 1
        ##  2) Rotational symmetry - the distance between 23:50 and 00:10 is
        ##      the same as the distance between 11:50 and 12:10, which is
        ##      lost with a straight-line representation of the time
        if self.time_of_day:
            hour = data["start_time"].dt.hour
            minute = data["start_time"].dt.minute
            total_minute = hour * 60 + minute
            _max_total_minute = 24 * 60 - 1

            output["time_of_day_sin"] = sin(total_minute * 2 * pi / _max_total_minute)
            output["time_of_day_cos"] = cos(total_minute * 2 * pi / _max_total_minute)
            time_columns.extend(["time_of_day_sin", "time_of_day_cos"])

        # Create time-of-week feature
        # We do the same here, using sin and cos. This way, the distance from
        # Sunday to Monday is the same as the distance from Friday to Saturday
        if self.time_of_week:
            day = data["start_time"].dt.day_of_week
            output["day_sin"] = sin(day * 2 * pi / 6)
            output["day_cos"] = cos(day * 2 * pi / 6)
            time_columns.extend(["day_sin", "day_cos"])

        # Create time-of-year feature
        # We do the same here as well, using sin and cos
        # Note that we only convert the day, as we do not need the granularity
        # of the actual time
        if self.time_of_year:
            year_day = data["start_time"].dt.day_of_year
            output["time_of_year_sin"] = sin(year_day * 2 * pi / 365)
            output["time_of_year_cos"] = cos(year_day * 2 * pi / 365)
            time_columns.extend(["time_of_year_sin", "time_of_year_cos"])

        # Create lag features
        shifted_columns = []
        # One day = 24 hrs/day * 60 min/hr / 5 min data
        one_day = int(24 * 60 / 5)
        # Yesterday's output
        if self.last_day_y:
            output["yesterday_y"] = output["target"].shift(one_day)
            shifted_columns.append("yesterday_y")

        # Two days ago's output = target shifted 2 * 24 * 60 / 5
        if self.two_last_day_y:
            output["2_yesterday_y"] = output["target"].shift(2 * one_day)
            shifted_columns.append("2_yesterday_y")
        
        # Add minor random value to last y, to decrease its importance for 
        # predictions. Use the post-clamped std and mean (see 
        # exploration/data_preprocessing.ipynb ) to disturb data nicely
        if self.randomize_last_y:
            rng = np.random.default_rng()
            std = 320       # from exploration/data_preprocessing.ipynb 
            mean = 8.9      # from exploration/data_preprocessing.ipynb
            random_values = 0.5 * std * rng.random(len(output["last_y"])) + mean
            output["last_y"] = output["last_y"] + random_values
        
        # Change order of columns to get similar Tensors
        column_order = (
            self.keep_columns
            + self.transform_columns
            + time_columns
            + shifted_columns + ["last_y", "target"]
        )
        output = output[column_order]


        # Remove last rows of data, due to shifting
        # Important note: We can safely do this without messing up the loading
        # of our data, as we know two things:
        #   - We had no missing data from the beginning
        #   - We only "removed" data from the end, when shifting the y-columns
        # When we dropna, then we simply remove the last N rows, where N is
        # changed depending on which preprocessing columns we keep
        output.dropna(axis=0, inplace=True)
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
    
    
    def reverse_y(self, y_column: pd.Series) -> pd.Series:
        return self.y_transformer.reverse(y_column)


if __name__ == "__main__":
    preprocesser = Preprocesser()
    data = pd.read_csv(DATA_PATH / "train.csv", parse_dates=["start_time"])
    print("DATA")
    print(data.describe())

    processed = preprocesser.fit_transform(data)
    print("FIT_TRANSFORM")
    print(processed.describe())
