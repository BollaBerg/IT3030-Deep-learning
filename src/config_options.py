from enum import Enum

class Loss(Enum):
    CROSS_ENTROPY = 0
    MSE = 1


class WeightRegularization(Enum):
    NONE = 0
    L1 = 1
    L2 = 2