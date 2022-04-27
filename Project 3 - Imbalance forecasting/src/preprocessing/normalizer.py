import pandas as pd

class Normalizer:
    is_fitted = False

    def fit(self, data: pd.Series):
        self.min = data.min()
        self.max = data.max()
        self.is_fitted = True

    def transform(self, data: pd.Series) -> pd.Series:
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transforming data")
        
        transformed_data = data.copy()
        return (transformed_data - self.min) / (self.max - self.min)

    def fit_transform(self, data: pd.Series) -> pd.Series:
        self.fit(data)
        return self.transform(data)
    
    def reverse(self, transformed_data: pd.Series) -> pd.Series:
        reversed_data = transformed_data.copy()
        return reversed_data * (self.max - self.min) + self.min