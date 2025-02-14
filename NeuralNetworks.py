import Layers as nn
import LossFunctions as lss

class NeuralNetworks():

    def __init__(self, layers:list[nn.Layer], depth):
        self.Layers = layers
        self.depth = depth

    def forward(self, x):
        for L in self.Layers:
            x = L.forward(x)
        return x

    def Train(self, x, y, lossFunc:lss.LossFunc, lr=10e-3):
        # First have a forward pass
        inputs = [x]
        for l in self.Layers:
            inputs.append(l.forward(inputs[-1]))

        FinalLoss = lossFunc.forward(inputs[-1], y)
        grad = lossFunc.grad(inputs[-1], y)
        print(f"Loss: {FinalLoss}")
        # Followed by a backward pass
        for i in range(self.depth-1, -1, -1):
            b, W, grad = self.Layers[i].backProp(inputs[i-1], grad)
            self.Layers[i].UpdateWeights(lr*W, lr*b)
        

