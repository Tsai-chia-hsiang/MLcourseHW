import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from NN.Layers.Activation import tanh_activation
from NN.Layers.Linear import Linear
from NN.Layers.Transform import BatchNormalization
from NN.Loss.MSE import MSEloss
from NN.Model import NNmodel
from NN.Optimizer.SGD import SGDopt
from NN.Optimizer.Ada import AdaGrad


def plot_change(l, names:list, saveto:os.PathLike, show=False):
    plt.figure(dpi=800)
    for li, name in zip(l, names):
        plt.plot(np.arange(len(li)), li, label=name)
    
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(saveto)
    if show:
        plt.show()
    plt.close()


class testNN(NNmodel):
    
    def __init__(self, in_dim, out_dim, transition:list) -> None:
        super().__init__()
        trans = [in_dim] + transition + [out_dim]
        for i in range(len(trans)-1):
            self.Sequence.append(Linear(in_dim=trans[i], out_dim=trans[i+1]))
            if i < len(trans)-2 :
                self.Sequence.append(BatchNormalization())  
                self.Sequence.append(tanh_activation())


def train(
    x_train:np.ndarray, y_train:np.ndarray, 
    x_val:np.ndarray, y_val:np.ndarray, 
    layer_trainsition =  [], pretrain=None,
    traininghyp={'epochs':100, 'bsize':32},
    opthyp = {'lr':0.001},
    modelsave=['model', 'model_0']
)->NNmodel:
    
    layer_transition = layer_trainsition

    model = testNN(
        x_train.shape[1], y_train.shape[1], 
        transition=layer_transition
    )
    model.describe()
    if pretrain is not None:
        model.loadmodel(pretrain)
    
    mseloss = MSEloss()
    optr = AdaGrad(
        model=model, lr=opthyp['lr'], momentum=opthyp['momentum'], 
        weight_decay=opthyp['weight_decay']
    )
    train_loss = []
    val_loss = []
    bar = tqdm(range(traininghyp['epochs']))
    bsize = traininghyp['bsize']
    bestmse = np.inf
    for e in bar:

        for b in range(0,x_train.shape[0], bsize):
            yhat = model.forward(x_train[b:b+bsize, :])
            mse = mseloss(yhat=yhat, y=y_train[b:b+bsize, :])
            optr.backward(loss = mseloss, batch_notation=True)
            optr.update()
        
        all_mse = mseloss(yhat=model.forward(x_train, req_g=False), y=y_train, req_g=False)
        train_loss.append(all_mse)
        val_mse = mseloss(yhat=model.forward(x_val, req_g=False), y=y_val, req_g=False)
        val_loss.append(val_mse)
        bar.set_postfix_str(
            f"train:{all_mse:.5f} |val: {val_mse:.5f} | best:{bestmse:.5f}"
        )
        
        if val_mse < bestmse:
            model.savemodel(
                os.path.join(modelsave[0], modelsave[1])
            )
            bestmse = val_mse
    
    bestmodel = testNN(
        x_train.shape[1], y_train.shape[1], 
        transition=layer_transition
    )
    bestmodel.loadmodel(
        os.path.join(modelsave[0],f"{modelsave[1]}.npz")
    )
 
    plot_change(
        l=[train_loss, val_loss], names=['train', 'val'],
        saveto=os.path.join(modelsave[0], f"loss_{modelsave[1]}.jpg"),show=False
    )
    return bestmodel, val_loss

def test(x_test, y_test, model:NNmodel)->float:
    Loss = MSEloss()
    y_pred = model.forward(x_test, req_g=False)
    mse = Loss(y=y_test, yhat=y_pred, req_g=False)
    return mse 

if __name__ == "__main__":
    
    triandir = os.path.join("gendata", "train")
    valdir = os.path.join("gendata", "val")
    testdir = os.path.join("gendata", "test")

    x_train = np.load(os.path.join(triandir, "x.npy"))
    y_train = np.load(os.path.join(triandir, "y.npy"))
    x_val = np.load(os.path.join(valdir, "x.npy"))
    y_val = np.load(os.path.join(valdir, "y.npy"))
    x_test = np.load(os.path.join(testdir, "x.npy"))
    y_test = np.load(os.path.join(testdir, "y.npy"))

    
    pretrain = os.path.join('model', 'random_para.npz')
    modelsavedir=os.path.join('model')

    bestmodel1, v1 = train(
        x_train=x_train, y_train=y_train, 
        x_val=x_val, y_val=y_val,
        opthyp={
            'lr':0.15, 'momentum':None, 
            'weight_decay':0.1
        }, 
        layer_trainsition = [16, 16, 8, 8, 4, 4],
        pretrain=pretrain,
        modelsave = [modelsavedir, 'm0']
    )
    bestmodel2, v2 = train(
        x_train=x_train, y_train=y_train, 
        x_val=x_val, y_val=y_val, 
        opthyp={
            'lr':0.15, 'momentum':0.9, 
            'weight_decay':0.1
        },
        layer_trainsition = [16, 16, 8, 8, 4, 4],
        pretrain=pretrain,
        modelsave = [modelsavedir, 'm09']
    )

    plot_change(
        l=[v1,v2], names=['SGD','momentum = 0.9'],
        saveto=os.path.join(modelsavedir,'cmp')
    )
    msetest1 = test(x_test=x_test ,y_test=y_test, model=bestmodel1)
    msetest2 = test(x_test=x_test, y_test=y_test, model=bestmodel2)
    print(f"test mse : mom None: {msetest1:.5f} ; m 0.9: {msetest2:.5f}")
