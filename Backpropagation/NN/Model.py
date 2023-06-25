import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))
from Loss.loss import _loss

class NNmodel():
    
    def __init__(self) -> None:
        self.Sequence = []

    def forward(self, x:np.ndarray, req_g=True, **kwargs)->np.ndarray:
        
        xt = x
        for L in self.Sequence:
            xt = L(xt, require_grad=req_g, **kwargs)
        
        return xt
    
    def model_Layers(self)->list:
        return list(L._name for L in self.Sequence)
    
    def describe(self):
        
        for L in self.model_Layers():
            if "Linear" in L:
                print()
            print(L, end=" -> ")
        
        print("Output")
    
    def savemodel(self, saveat:os.PathLike):
        
        weight= []
        
        for li in range(len(self.Sequence)):
            if hasattr(self.Sequence[li], 'load_pretrain'):
                weight.append(self.Sequence[li]._weight)
        
        np.savez(saveat, *weight)

    def loadmodel(self, loadfrom:os.PathLike):
        
        ws = np.load(loadfrom)
        i = 0
        for li in range(len(self.Sequence)):
            if hasattr(self.Sequence[li], 'load_pretrain'):
                self.Sequence[li].load_pretrain(w=ws[f'arr_{i}'])
                i += 1

def test(xtest:np.ndarray, ytest:np.ndarray, model:NNmodel, loss:type[_loss])->float:
    Loss = loss()
    return Loss(
        yhat=model.forward(xtest, req_g=False), y=ytest, 
        req_g=False
    )