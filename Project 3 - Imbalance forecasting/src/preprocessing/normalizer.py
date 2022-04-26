import pandas as pd

class Normalizer:
    is_fitted = False

    def fit(self, data: pd.Series):
        self.mean = data.mean()
        self.std = data.std()
        self.is_fitted = True

    def transform(self, data: pd.Series) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transforming data")
        
        transformed_data = data.copy()
        return (transformed_data - self.mean) / self.std

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)