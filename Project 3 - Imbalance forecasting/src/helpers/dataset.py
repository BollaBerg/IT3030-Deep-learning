from torch import Tensor
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    """
    Dataset class, adapted for time series data

    Returns a window of data, instead of the singular data points.
    Inspired by 
        https://discuss.pytorch.org/t/dataloader-for-a-lstm-model-with-a-sliding-window/22235/2

    """
    def __init__(self, inputs: Tensor, targets: Tensor, window: int):
        self.inputs = inputs
        self.targets = targets
        self.window = window
    
    def __getitem__(self, index: int):
        inputs = self.inputs[index: index + self.window]
        target = self.targets[index + self.window]
        return inputs, target
    
    def __len__(self):
        return len(self.inputs) - self.window