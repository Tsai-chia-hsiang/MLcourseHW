import numpy as np
from scipy.special import expit
import sys
import os
sys.path.append(os.path.dirname(__file__))
from Layer import _Layer

class tanh_activation(_Layer):

    def __init__(self) -> None:
        super().__init__()
        self._type = "act"
        self._name = "tanh"
    
    def  __call__(self, x:np.ndarray, require_grad=False, **kwargs) ->np.ndarray:
        super().forward(x=x, require_grad=require_grad)
        return np.tanh(x)
    
    def _gradiant_func(self, x:np.ndarray)->np.ndarray:
        return (1.0-((np.tanh(x))**2))
    
class sigmoid_activation(_Layer):

    def __init__(self) -> None:
        super().__init__()
        self._type = "act"
        self._name = "sigmoid"
    
    def _sigmoid_cal(self, x):
        return 1.0/(1.0+expit(-x))

    def  __call__(self, x:np.ndarray, require_grad=False, **kwargs) ->np.ndarray:
        super().forward(x=x, require_grad=require_grad)
        return self._sigmoid_cal(x)
    
    def _gradiant_func(self, x:np.ndarray)->np.ndarray:
        return self._sigmoid_cal(x)*(1.0-self._sigmoid_cal(x))

class ReLu_activation(_Layer):

    def __init__(self) -> None:
        super().__init__()
        self._type = "act"
        self._name = "relu"
    

    def  __call__(self, x:np.ndarray, require_grad=False, **kwargs) ->np.ndarray:
        super().forward(x=x, require_grad=require_grad)
        return np.maximum(0.0, x)
    
    def _gradiant_func(self, x:np.ndarray)->np.ndarray:
        g =(x>0).astype(np.float32)
        return g