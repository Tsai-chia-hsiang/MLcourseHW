import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from Layer import _Layer

class Linear(_Layer):
    
    def __init__(self,in_dim,out_dim, bias = True) -> None:
        
        super().__init__()
        self._type = "Linear"
        self._req_bias = bias
        ind = in_dim
        if self._req_bias:
            ind += 1
        
        self._weight = np.random.randn(ind, out_dim)
        self._name = f"Linear({in_dim}+{int(self._req_bias)}, {out_dim})"
        
    def __call__(self, x:np.ndarray, require_grad = False, **kwargs)->np.ndarray:
        
        x_ = np.hstack((x, np.ones((x.shape[0], 1), dtype=np.float32))) \
            if self._req_bias else x.copy()
        
        super().forward(x_, require_grad=require_grad)
        
        return x_@self._weight
    
    def _gradiant_func(self, x:np.ndarray)->np.ndarray:
        return x.T
    
    def load_pretrain(self, w:np.ndarray):
        self._weight = w.copy()

    def get_parameters(self):
        return self._weight.copy()
    