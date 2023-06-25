import numpy as np
from tqdm import tqdm
import copy
import os
from NN.Layers.Activation import tanh_activation
from NN.Layers.Linear import Linear
from NN.Layers.Transform import BatchNormalization
from NN.Loss.MSE import MSEloss
from NN.Model import NNmodel, test
from NN.Optimizer.optr import _optimizer
from NN.Optimizer.SGD import SGDopt
from NN.Optimizer.Ada import AdaGrad
from vis.plotchange import plot_changes

class testNN(NNmodel):
    
    def __init__(self, in_dim, out_dim, transition:list) -> None:
        
        super().__init__()
        
        trans = [in_dim] + transition + [out_dim]
        
        for i in range(len(trans)-1):
            self.Sequence.append(Linear(in_dim=trans[i], out_dim=trans[i+1]))
            if i < len(trans)-2 :
                self.Sequence.append(BatchNormalization())  
                self.Sequence.append(tanh_activation())

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
    xtrain:np.ndarray, ytrain:np.ndarray, 
    xval:np.ndarray, yval:np.ndarray, 
    modelframe:type[NNmodel], modelarch:list, opt:type[_optimizer], 
    hyp:dict, modelsavepath:os.PathLike,
    epochs=100, bsize = 32,loadpretrain:os.PathLike=None
)->tuple:
    
    model = modelframe(
        in_dim=xtrain.shape[1], out_dim=ytrain.shape[1], 
        transition=modelarch
    )
    if loadpretrain is not None:
        model.loadmodel(loadpretrain)
    
    Loss = MSEloss()

    trainLoss = [] ; valLoss = []
    optimizer = opt(
        model=model, lr=hyp['lr'],
        **{k:v for k,v in hyp.items() if k!= 'lr'}
    )
    
    pbar = tqdm(range(epochs))
    minmse = np.inf
    for e in pbar:
        
        #A batch
        trainmse = 0
        for b in range(0,xtrain.shape[0], bsize):
            
            yhat = model.forward(x=xtrain[b:b+bsize, :])
            trainmse += Loss(yhat=yhat, y=ytrain[b:b+bsize, :])

            optimizer.backward(loss=Loss, batch_notation=True)
            optimizer.update()

        trainLoss.append(trainmse/(xtrain.shape[0]/bsize))
        
        valmse = Loss(
            yhat=model.forward(x=xval, req_g=False), 
            y=yval, req_g=False
        )
        valLoss.append(valmse)
        if valmse < minmse:
            model.savemodel(saveat=modelsavepath)
            minmse = valmse

    bestmodel = modelframe(
        in_dim=xtrain.shape[1], out_dim=ytrain.shape[1], 
        transition=modelarch
    )
    bestmodel.loadmodel(f"{modelsavepath}.npz")
    return bestmodel, trainLoss, valLoss



if __name__ == "__main__":

    xtrain, ytrain, xval, yval, xtest, ytest  = loaddata(os.path.join("gendata"))
    opt_type = [SGDopt, SGDopt, AdaGrad]
    hyp = [{'lr':0.001, 'weight_decay':0.1},
           {'lr':0.001, 'momentum':0.9, 'weight_decay':0.1},
           {'lr':0.15, 'weight_decay':0.1}
        ]
    model_name = ["sgd","sgd_mom09", "ada"]
    trainlosses = []
    vallossses = []
    for opti, hypi, name in zip(opt_type, hyp, model_name):
        print(name)
        bmodel, trainloss, valloss = train(
            xtrain, ytrain, xval, yval,
            modelframe=testNN, modelarch=[16, 16, 8, 8, 4, 4],
            opt=opti, hyp=hypi,
            modelsavepath=os.path.join("model", name),
            loadpretrain=os.path.join("model","random_para.npz" )
        )
        testLoss = test(xtest, ytest, bmodel, MSEloss)
        print(testLoss)
        trainlosses.append(trainloss)
        vallossses.append(valloss)

    plot_changes(
        l=trainlosses, name=model_name, title="Train loss",
        saveto="trainloss.jpg"
    )
    plot_changes(
        l=vallossses, name=model_name, title="val loss",
        saveto="valloss.jpg"
    )
