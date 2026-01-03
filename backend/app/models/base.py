from abc import ABC, abstractmethod
import numpy as np

class SuperResolutionModel(ABC):
    @abstractmethod
    def run(self, image: np.ndarray) -> np.ndarray:
        pass

    