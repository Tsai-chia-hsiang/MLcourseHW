import numpy as np

class _Layer():
    
    def __init__(self) -> None:
        self.gradiant = None
    
    def forward(self, x:np.ndarray, require_grad=False)->np.ndarray:
        if require_grad:
            self.gradiant = self._gradiant_func(x)
    