import numpy as np
from abc import ABC, abstractmethod

class LossFunc(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def singleForward(self, x, y):
        pass

    @abstractmethod
    def forward(self, x, y):
        pass


class LMSError(LossFunc):

    def __init__(self):
        pass

    def singleForward(self, x, y):
        return ((x - y)**2)/2

    def forward(self, x, y):
        return np.mean((x - y)**2)/2

