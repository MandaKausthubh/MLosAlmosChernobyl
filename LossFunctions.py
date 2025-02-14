import numpy as np
from abc import ABC, abstractmethod

class LossFunc(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def singleForward(self, prediction, true) -> np.float32:
        pass

    @abstractmethod
    def forward(self, prediction, true) -> np.float32:
        pass

    @abstractmethod
    def grad(self, pred, true) -> np.ndarray:
        pass

class LMSError(LossFunc):

    def __init__(self):
        pass

    def singleForward(self, prediction, true):
        return ((prediction - true)**2)/2

    def forward(self, prediction, true):
        return np.mean((prediction - true)**2)/2

    def grad(self, pred, true):
        return (pred - true)


