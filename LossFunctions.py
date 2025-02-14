import numpy as np
from abc import ABC, abstractmethod

class LossFunc(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def singleForward(self, prediction, true):
        pass

    @abstractmethod
    def forward(self, prediction, true):
        pass

    @abstractmethod
    def grad(self, pred, true):
        pass

class LMSError(LossFunc):

    def __init__(self):
        pass

    def singleForward(self, prediction, true):
        return ((prediction - true)**2)/2

    def forward(self, prediction, true):
        return np.mean((prediction - true)**2)/2

    def grad(self, pred, true):
        return np.mean(pred - true, axis=0)


