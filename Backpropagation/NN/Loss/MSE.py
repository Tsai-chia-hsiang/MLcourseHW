import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from loss import _loss

class MSEloss(_loss):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, yhat:np.ndarray, y:np.ndarray, req_g=True) -> np.float32:
        super().__call__(req_g, yhat, y)
        return np.mean((yhat-y)**2)/2 

    def _gradiant_func(self, yhat, y) -> np.ndarray:
        self.gradiant = yhat-y
    