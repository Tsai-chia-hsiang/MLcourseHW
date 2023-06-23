
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from NN.Model import NNmodel
from optr import _optimizer


class SGDopt(_optimizer):
    
    def __init__(self, model: NNmodel, lr=0.001,**kwargs) -> None:
        
        super().__init__(model, lr, **kwargs)
        self._momentum = kwargs['momentum'] if 'momentum' in kwargs else None
        self._lasttime_update = []
        self._history = 0
    

    def update(self):
        linear_idx = 0
        for li in range(len(self._model.Sequence)):
            
            if self._model.Sequence[li]._type == "Linear":
                thisupdate = self._lr*(self._model.Sequence[li].gradiant)
                self._model.Sequence[li]._weight -= thisupdate
            
                if self._momentum is not None :
                    
                    if self._history:
                            
                        self._model.Sequence[li]._weight -= \
                        self._momentum*self._lasttime_update[linear_idx]
                        self._lasttime_update[linear_idx] = thisupdate
                        linear_idx += 1
                    
                    else :
                        #first time update, no history 
                        self._lasttime_update.append(thisupdate)

        if self._history == 0:
            self._history = 1