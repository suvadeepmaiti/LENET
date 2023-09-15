from layers import BaseLayer, Conv2D, SubSample, Dense, RBF, Activation
# !pip3 install tabulate
from tabulate import tabulate
import numpy as np

# Defines the architecture of the lenet model
class Lenet_SMAI(object):

    def __init__(self, input_shape = (32, 32, 1), name = 'Lenet') -> None:
        super().__init__()
        assert input_shape != None
        self.name = name
        self.layers = [
            Conv2D(6, (5,5)),
            Activation('relu'),
            SubSample(2),
            Conv2D(16, (5,5)),
            Activation('relu'),
            SubSample(2),
            Conv2D(120, (5,5)),
            Activation('tanh'),
            Dense(84),
            Activation('tanh'),
            RBF(10)
        ]
        self.input_shape = input_shape
        prev_input_shape = input_shape
        for layer in self.layers:
           prev_input_shape = layer.init_layer(prev_input_shape).out_shape

     
    def summary(self):
        # prints information of inputs, outputs and parameters of the model
        print(f"\t\t-------{self.name}-------\t\t")
        table = []
        total = 0
        for layer in self.layers:
            t = self.total_params(layer.params)
            table.append((layer.in_shape, str(layer), layer.out_shape, t))
            total += t
        print(tabulate(table, headers=["in", "Name (weight, bias) ", "out", "total_params"], tablefmt="psql"))
        print(f"Total Number of parameters = {total}")
        
    def total_params(self, params : 'list(tuple)') -> int:
        if not params: return 0
        return np.sum([np.prod(e.shape) for e in params])

    def compile_adam(self, b1= 0.9, b2 = 0.999, epsilon = 1e-8, eta = 0.01):
        # registers each layer to adam optimizer 
         self.optimizer = 'adam'
         for l in self.layers:
            if isinstance(l, BaseLayer): l.compile_adam(b1= b1, b2 = b2, epsilon = epsilon, eta = eta)


    def __call__(self, input, label=None, mode = 'train'):
        # runs forward propagation in train or test mode
        o = input
        for layer in self.layers:
            if isinstance(layer, RBF): o = layer(o, label, mode)
            else: o = layer(o)
        return o

    def compute_gradients(self):
        # obtains gradient from each layer and passes it backward
        next_d = 1
        grads = {}
        for layer in reversed(self.layers):
            dW, db, next_d = layer.__gradients__(next_d)
            if dW is None and db is None: continue
            # upadating if eligible
            grads[layer] = (dW, db)
        return grads

    def apply_gradients(self, gradients):
        # on all layers which have registered an optimizer and have parameters update weights 
        for k,v in gradients.items():
            dW, db = v
            if isinstance(k, BaseLayer): k.optimzers[self.optimizer](dW , db)