import numpy as np
import Activations as act
import NeuralNetworks as nn
import LossFunctions as ls
import Layers as lay

x = np.random.rand(100, 5)
y = np.random.rand(100, 2)

l1 = lay.Layer(5, 10, act.ReLu())
l2 = lay.Layer(10, 10, act.ReLu())
l3 = lay.Layer(10, 2, act.ReLu())

N = nn.NeuralNetworks([l1, l2, l3], 3)
print(N.forward(x).shape)

lossFunc = ls.LMSError()
N.Train(x, y, lossFunc)
