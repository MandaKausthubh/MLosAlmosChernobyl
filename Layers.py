import numpy as np
import Activations as act
import LossFunctions

class Layer():

    def __init__(self, input_size:int, output_size:int,activation:act.Activation) -> None:
        self.inp, self.out = input_size, output_size
        self.weights = np.random.rand(self.out, self.inp)
        self.biases = np.random.rand(self.out, 1)
        self.activation = activation
        pass



    def forward(self, x):
        print(f"x:{x.shape}, w:{self.weights.shape}, b:{self.biases.shape}")
        return self.activation.forward(((self.weights @ x.T) +
            self.biases @ np.ones((1,len(x)))).T)



    def UpdateWeights(self, W_delta, b_delta):
        self.weights += W_delta
        self.biases += b_delta


    def backProp(self, x, grad):
        z = ((self.weights @ x.T) + self.biases @ np.ones((1,len(x)))).T
        G = self.activation.Differential(z) * grad
        print(G.shape)
        # Calculate delta b
        del_b = np.mean(G, axis=0).reshape(-1,1)
        # Calculate delta W
        del_W = np.mean(np.array([
            G[p].reshape(-1, 1) @ x[p].reshape(1, -1)
            for p in range(len(x))
        ]), axis=0)
        # Calculate delta a
        del_a = np.array([
            G[p].reshape(1,-1) @ self.weights
            for p in range(len(x))
        ]).reshape(len(x), -1)
        return del_b, del_W, del_a



