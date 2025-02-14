import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def forward(self, x) -> np.ndarray:
        pass

    @abstractmethod
    def Differential(self, x) -> np.ndarray:
        pass



class ReLu(Activation):

    def __init__(self) -> None:
        super().__init__()

    def singleForward(self, x):
        return np.array([y if y >= 0 else 0 for y in x])

    def forward(self, x) -> np.ndarray:
        return np.array([self.singleForward(y) for y in x])

    def singleDifferential(self, x):
        return np.array([1 if y >= 0 else 0 for y in x])

    def Differential(self, x) -> np.ndarray:
        return np.array([self.singleDifferential(y) for y in x])



