import numpy as np

class BatchNormalization():
    
    def __init__(self, beta=0, gamma=1.0) -> None:
    
        self.__b = beta
        self.__g = gamma
        self._type="Trans"
        self.__eps = 1e-12
        self._name = f"BatchNorm"

    def _expand(self, x, wantshape):
        return np.tile(x, wantshape)

    def  __call__(self, x, **kwargs):
        mu = self._expand(np.mean(x, axis=0), wantshape=(x.shape[0], 1))
        variance = self._expand(np.mean((x-mu)**2, axis=0),  wantshape=(x.shape[0], 1))
        zscore = (x - mu)/(variance + self.__eps)
        return  self.__g*zscore + self.__b