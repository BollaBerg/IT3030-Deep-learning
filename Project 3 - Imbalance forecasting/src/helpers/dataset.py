from torch import Tensor
from torch.utils.data import Dataset, T_co

class TimeSeriesDataset(Dataset):
    """
    Dataset class, adapted for time series data

    Returns a window of data, instead of the singular data points.
    Inspired by 
        https://discuss.pytorch.org/t/dataloader-for-a-lstm-model-with-a-sliding-window/22235/2

    """
    def __init__(self, data: Tensor, window: int):
        self.data = data
        self.window = window
    
    def __getitem__(self, index: int) -> T_co:
        return self.data[index: index + self.window]
    
    def __len__(self):
        return len(self.data) - self.window