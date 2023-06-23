
import numpy as np

class _loss():
    
    def __init__(self) -> None:
        self.gradiant:np.ndarray=None
    
    def __call__(self, require_grad=True, *args):
        if require_grad:
            self._gradiant_func(*args)

