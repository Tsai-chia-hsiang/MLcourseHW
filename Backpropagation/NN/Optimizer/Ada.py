import sys
import os
from NN.Loss.loss import _loss
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from NN.Model import NNmodel
from optr import _optimizer


class AdaGrad(_optimizer):
    
    def __init__(self, model: NNmodel, lr=0.01, **kwargs) -> None:
    
        super().__init__(model, lr, **kwargs)
        self._approx_2nderiv = []
        self._history = 0

    def update(self):

        linear_idx = 0
        for li in range(len(self._model.Sequence)):
            
            if self._model.Sequence[li]._type == "Linear":
                
                if self._history == 0:
                    self._approx_2nderiv.append(
                        (self._model.Sequence[li].gradiant)**2
                    )
                    thisupdate = self._lr*(self._model.Sequence[li].gradiant)
                else:
                    thisupdate = self._lr/(
                        self._approx_2nderiv[linear_idx]**0.5 + 1e-8
                    )*(self._model.Sequence[li].gradiant)
                    self._approx_2nderiv[linear_idx] += (self._model.Sequence[li].gradiant)**2
                    
                self._model.Sequence[li]._weight -= thisupdate
                linear_idx += 1

        if self._history == 0:
            self._history = 1