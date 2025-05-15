import random
from gradient import Value

class Neuron:
    # nin -> Number of neurons
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    # nin -> number of layers, nout -> number of neurons in each layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    # nin -> number of layers, nouts -> list of the sizes of layers we want in the MLP
    def __init__(self, nin, nouts):
        sz = [nin] + nouts 
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __zero_grad__(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def optimize(self, learning_rate, cycles, inp, out):
        for k in range(cycles):
            # forward pass
            ypred = [self.__call__(x) for x in inp]
            loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, out))
            # zero grad the gradients
            self.__zero_grad__()
            # backward propagtion
            loss.backward()
            for p in self.parameters():
                # update (gradient descent)
                p.data += -learning_rate * p.grad

        return ypred, loss