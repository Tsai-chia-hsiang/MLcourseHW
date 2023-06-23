import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from NN.Model import NNmodel
from Loss.loss import _loss

class _optimizer():
    
    def __init__(self, model:NNmodel, lr= 0.01, **kwargs) -> None:
        
        self._model = model
        self._lr = lr
        self._deltai = None
        self._bpmethod = {
            'Linear':self._Linear_bp,
            'act':self._act_bp
        } 
        self. _weight_decay = kwargs['weight_decay'] if \
            'weight_decay' in kwargs else 0

    def backward(self, loss:_loss, batch_notation=False):

        self._deltai = loss.gradiant.copy()
        for li in range(len(self._model.Sequence)-1, -1, -1):
            ltype = self._model.Sequence[li]._type
            if ltype in self._bpmethod:
                self._bpmethod[ltype](li=li, batch = batch_notation)

    def _Linear_bp(self, li, batch=False):
       
        m = self._model.Sequence[li].gradiant.shape[1]
        bsize = m if batch else 1.0

        self._model.Sequence[li].gradiant = (
            (self._model.Sequence[li].gradiant/bsize)@self._deltai
        ) 
        if self._weight_decay:
            self._model.Sequence[li].gradiant += \
            (self._weight_decay/m)*self._model.Sequence[li]._weight

        self._deltai = self._deltai@(self._model.Sequence[li]._weight.T)
    
    def _act_bp(self, li, **kwargs):

        previous_d = self._deltai.shape
        current_d = self._model.Sequence[li].gradiant.shape
        if  previous_d[1] == current_d[1] + 1 :
        
            ## previous having a bias term, no need to bp
            self._deltai = self._deltai[:, :-1]
        
        self._deltai *= self._model.Sequence[li].gradiant
    
    def update(self):
        pass
            
