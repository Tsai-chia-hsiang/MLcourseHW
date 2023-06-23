import numpy as np
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
    