import numpy as np
from tqdm import tqdm
import copy
import os
from NN.Layers.Activation import tanh_activation, sigmoid_activation, ReLu_activation
from NN.Layers.Linear import Linear
from NN.Layers.Transform import BatchNormalization
from NN.Loss.loss import _loss
from NN.Loss.MSE import MSEloss
from NN.Model import NNmodel, test
from NN.Optimizer.optr import _optimizer
from NN.Optimizer.SGD import SGDopt
from NN.Optimizer.Ada import AdaGrad
from vis.plotchange import plot_changes

def makepath(p:os.PathLike)->os.PathLike:
    if not os.path.exists(p):
        os.mkdir(p)
    return p

class testNN(NNmodel):
    
    def __init__(self, in_dim, out_dim, transition:list) -> None:
        
        super().__init__()
        
        trans = [in_dim] + transition + [out_dim]
        
        for i in range(len(trans)-1):
            self.Sequence.append(Linear(in_dim=trans[i], out_dim=trans[i+1]))
            if i < len(trans)-2 :
                self.Sequence.append(BatchNormalization())  
                self.Sequence.append(ReLu_activation())

def loaddata(root:os.PathLike):
    
    triandir = os.path.join(root, "train")
    valdir = os.path.join(root, "val")
    testdir = os.path.join(root, "test")

    xtrain = np.load(os.path.join(triandir, "x.npy"))
    ytrain = np.load(os.path.join(triandir, "y.npy"))
    xval = np.load(os.path.join(valdir, "x.npy"))
    yval = np.load(os.path.join(valdir, "y.npy"))
    xtest = np.load(os.path.join(testdir, "x.npy"))
    ytest = np.load(os.path.join(testdir, "y.npy"))

    return xtrain, ytrain, xval, yval, xtest, ytest 

def train(
    xtrain:np.ndarray, ytrain:np.ndarray, xval:np.ndarray, yval:np.ndarray, 
    Model:NNmodel, LossFunction:type[_loss], opt:type[_optimizer], 
    hyp:dict, modelsavepath:os.PathLike, epochs=100, bsize = 32
)->tuple:
    
    model = copy.deepcopy(Model)
    Loss = LossFunction()
    optimizer = opt(
        model=model, lr=hyp['lr'],
        **{k:v for k,v in hyp.items() if k!= 'lr'}
    )
    trainLoss, valLoss= [] ,[]
    
    pbar = tqdm(range(epochs))
    minl = np.inf
    for _ in pbar:
        
        #A batch
        trainl = 0
        for b in range(0,xtrain.shape[0], bsize):
            
            yhat = model.forward(x=xtrain[b:b+bsize, :])

            trainl += Loss(yhat=yhat, y=ytrain[b:b+bsize, :])/(xtrain.shape[0]/bsize)

            optimizer.backward(loss=Loss, batch_notation=True)
            optimizer.update()
  
        
        trainLoss.append(trainl)
        
        vall = Loss(
            yhat=model.forward(x=xval, req_g=False), 
            y=yval, req_g=False
        )
        valLoss.append(vall)
        pbar.set_postfix_str(
            f"train:{trainl:.5f}, val:{vall:.5f}, bestva:{minl:.5f}"
        )
        if vall < minl:
            model.savemodel(saveat=modelsavepath)
            minl = vall

    model.loadmodel(f"{modelsavepath}.npz")
    return model, trainLoss, valLoss



def Comparsion_different_optimizer(
    sampledatadir:os.PathLike, cmp:dict, pretrain_path :os.PathLike,
    modelsaveroot:os.PathLike, resultsaveroot:os.PathLike
):

    xtrain, ytrain, xval, yval, xtest, ytest  = loaddata(sampledatadir)

    model = testNN(
        in_dim=xtrain.shape[1], out_dim=ytrain.shape[1], 
        transition=[16, 16, 8, 8, 4, 4]
    )
    model.loadmodel(pretrain_path)
    
    trainlosses = []
    vallossses = []
    for opti, hypi, name in zip(cmp['optr'], cmp['hyp'], cmp['model_name']):
        print(name)
        epochs = 100
        bmodel, trainloss, valloss = train(
            xtrain, ytrain, xval, yval,Model=model, 
            opt=opti, hyp=hypi, LossFunction=MSEloss,
            modelsavepath=os.path.join(modelsaveroot, name),
            epochs=epochs
        )
        testLoss = test(xtest, ytest, bmodel, MSEloss)
        print(f"testing loss : {testLoss:.5f}")
        trainlosses.append(trainloss)
        vallossses.append(valloss)

    plot_changes(
        l=trainlosses, name=cmp['model_name'], title="Train loss",
        saveto=os.path.join(resultsaveroot,"trainloss.jpg")
    )
    plot_changes(
        l=vallossses, name=cmp['model_name'], title="val loss",
        saveto=os.path.join(resultsaveroot,"valloss.jpg")
    )
 
if __name__ == "__main__":

    cmp_optr = {
        'optr':[SGDopt, SGDopt,AdaGrad],
        'hyp':[ {'lr':0.001, 'weight_decay':0.1},
            {'lr':0.001, 'momentum':0.9, 'weight_decay':0.1},
            {'lr':0.005, 'weight_decay':0.1}],
        'model_name':[ "sgd","sgd_mom09" ,"ada"]
    }

    Comparsion_different_optimizer(
        sampledatadir=os.path.join('gendata'),
        cmp=cmp_optr, pretrain_path=os.path.join("model", "random_para.npz"),
        modelsaveroot=os.path.join("model"),
        resultsaveroot=makepath(os.path.join("comparsion_optrs"))
    )

