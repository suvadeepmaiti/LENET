import numpy as np
from RBF_init import rbf_initialize

# usefful dictionary which defines f(x) and f'(x)
__builtin_funcs__ =  {
    'tanh' : (lambda x : 1.7159*np.tanh(2*x/3), lambda x : 1.14393*(1-np.power(np.tanh(2*x/3),2))),
    'mse' : (lambda x,y : np.mean(np.square(y - x)), lambda x,y : -np.mean(y - x, axis=1)),
    'sigmoid':(lambda x : 1 / (1 + np.exp(-x)), lambda x : np.exp(-x) / np.power((1+np.exp(-x)),2)),
    'relu' : (lambda x : np.maximum(x,0), lambda x : (x>0)*1 )
}

# initializes the weight and biases with the input shape, in normal distribution and uniform distribution respectively
def initialize(shape):
    mu, sigma = 0, 0.1
    b_shape = (1,1,1,shape[-1]) if len(shape) == 4 else (shape[-1],)
    weight = np.random.normal(mu, sigma,  shape)
    bias  = np.ones(b_shape)*0.01
    return weight, bias

# pads the input with zeros around the border
def zero_pad(input, pad):
    return np.pad(input, ((0, ), (pad, ), (pad, ), (0, )), 'constant', constant_values=(0, 0))   



""" *************** Layers *************** """

# class BaseLayer, provides its children an optimizer to register
class BaseLayer(object):
    def __init__(self) -> None:
        super().__init__()
        self.params = None, None
        self.optimzers =  {'adam' : None }

    # registering an optimizer
    def compile_adam(self, b1= 0.9, b2 = 0.999, epsilon = 1e-8, eta = 0.01):
        self.beta1, self.beta2, self.epsilon, self.eta = b1, b2, epsilon, eta
        self.m_dw, self.m_db, self.v_dw, self.v_db = None, None, None, None
        def update(dW, db, t = 1):
            # usual definition of adam optimizer with momentum beta1, beta2 and eta learning rate
            if self.m_dw is None or self.m_db is None or self.v_dw is None or self.v_db is None:
                self.m_dw, self.m_db, self.v_dw, self.v_db =  np.zeros(dW.shape) , np.zeros(db.shape), np.zeros(dW.shape), np.zeros(db.shape)
            if self.m_dw.shape != dW.shape or self.m_db.shape != db.shape:
                self.m_dw, self.m_db, self.v_dw, self.v_db =  np.zeros(dW.shape) , np.zeros(db.shape), np.zeros(dW.shape), np.zeros(db.shape)
            # running average of adam parameters
            self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dW
            self.m_db = self.beta1*self.m_db + (1-self.beta1)*db
            self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dW**2)
            self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)
            #normalizing adam parameters
            m_dw_corr = self.m_dw/(1-self.beta1**t)
            m_db_corr = self.m_db/(1-self.beta1**t)
            v_dw_corr = self.v_dw/(1-self.beta2**t)
            v_db_corr = self.v_db/(1-self.beta2**t)
            #weight, bias update
            w = self.params[0] - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
            b = self.params[1] - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
            #setting values
            self.params = (w, b) 
        # registering an optimizer
        self.optimzers['adam'] = update

# Convolutional Layer
class Conv2D(BaseLayer):
    
#     initialises the parameters num_filters, kernel_shape, stride, padding
    def __init__(self, num_filters, kernel_shape, stride = 1, padding = 0) -> None:
        super().__init__()
        self.cache = None
        self.in_shape, self.out_shape = None, None
        self.kernel_shape, self.num_filters = kernel_shape, num_filters
        self.params = None, None
        self.padding, self.stride = padding, stride
        
#     initialises the weights and biases according to above parameters
    def init_layer(self, in_shape):
        assert len(in_shape) == 3
        self.in_shape = in_shape
        self.out_shape = (in_shape[0] - self.kernel_shape[0] + 1, in_shape[1] - self.kernel_shape[1] + 1 , self.num_filters)
        self.param_shape = self.kernel_shape + (self.in_shape[-1], self.num_filters)
        self.params = initialize(self.param_shape)
        return self

#     forward pass
    def __call__(self, input):
        z = np.zeros((input.shape[0], ) + self.out_shape)
        input_pad = zero_pad(input, self.padding)
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                slice = input_pad[:, h*self.stride:h*self.stride+self.kernel_shape[0], w*self.stride:w*self.stride+self.kernel_shape[1], :]  # each input slice for (h,w) of the output
                z[:, h, w, :] = np.tensordot(slice, self.params[0], axes=([1,2,3],[0,1,2])) + self.params[1] # calculating the convolutionn choosing (1,2,3) axes as 0 is reserved for batch shape
        self.cache = input #caching input which is required in gradients
        return z

#     computes the gradients for back pass
    def __gradients__(self, next_d):
        dinput = zero_pad(np.zeros(self.cache.shape), self.padding)
        dW = np.zeros(self.params[0].shape) # kernel derivative
        db = np.zeros(self.params[1].shape) # kernel bias derivative
        cache_padded = zero_pad(self.cache, self.padding) # cache stores the input
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                s, e = (h*self.stride, h*self.stride+self.kernel_shape[0]), (w*self.stride, w*self.stride+self.kernel_shape[1])
                slice = cache_padded[:,s[0]:s[1], e[0]:e[1], :] # extracting each slice of input responsible for (h,w) derivates of the kernel
                dinput[:, s[0]:s[1], e[0]:e[1], :] += np.transpose(self.params[0] @ next_d[:, h, w, :].T, (3,0,1,2))
                dW += np.matmul(np.transpose(slice, (1,2,3,0)), next_d[:, h, w, :]) # (h, w, f, b) x (b, k) , summing the derivates of the kernel contributing to the slice which only depends of the input slice and the output cell derivatives
                db += np.sum(next_d[:, h, w, :], axis=0) # summing the derivatives of the bias
        return dW, db, dinput

    def __str__(self) -> str:
        return f"CONV2D{self.param_shape ,  self.params[1].shape}"

# A pooling layer, with trainable parameters
class SubSample(BaseLayer):

    def __init__(self, kernel_shape, stride = 2, padding = 0) -> None:
        super().__init__()
        self.cache = None
        self.kernel_shape = kernel_shape
        self.params = None, None
        self.padding, self.stride = padding, stride
        self.in_shape, self.out_shape = None, None
    
    def init_layer(self, in_shape):
        assert len(in_shape) == 3
        self.in_shape = in_shape
        self.out_shape = ( (in_shape[0] - self.kernel_shape) // self.stride + 1, (in_shape[1] - self.kernel_shape + 1) //self.stride + 1 , in_shape[-1] )
        self.params = np.random.normal(0, 0.1, (1,1,1,self.in_shape[-1])),  np.random.normal(0, 0.1, (1,1,1,self.in_shape[-1])) # param initialization
        return self

    def __call__(self, input):
        o = np.zeros((input.shape[0], ) + self.out_shape) 
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                slice = input[:, h*self.stride:h*self.stride+self.kernel_shape, w*self.stride:w*self.stride+self.kernel_shape, :] # computing a input slice [b , h1:h2 , w1:w2, f]
                o[:, h, w, :] = np.average(slice, axis=(1,2)) # average pooling, differs from direct summing in the paper
        self.cache = (input, o) # cache inputs and temporary output for backprop
        output = o * self.params[0] + self.params[1] # scaling subsample by weight and bias parameters
        return output

    def __gradients__(self, next_d):
        prev_input, out_ = self.cache # unwrapping input and subsample output
        db = np.mean(next_d, axis=(0,1,2), keepdims=True) # gradient of bias parameter
        dW = np.mean(np.multiply(next_d, out_), axis = (0,1,2), keepdims=True) # gradient of weight parameter
        next_d_after = next_d * self.params[0] # gradient wrt subsample output
        dinput = np.zeros(prev_input.shape)
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                s , e = (h*self.stride, h*self.stride+self.kernel_shape), (w*self.stride, w*self.stride+self.kernel_shape)
                # extracting gradient of subsample output cell
                da = next_d_after[:, h, w, :][:,np.newaxis,np.newaxis,:]
                # computing gradient for each input contributing to the above subsample output ( logic is to expand and average the contributions for brevity and simplicity ) 
                dinput[:, s[0]: s[1], e[0]: e[1], :] += np.repeat(np.repeat(da, self.kernel_shape, axis=1), self.kernel_shape, axis=2)/(self.kernel_shape * self.kernel_shape)
        return dW, db, dinput

    def __str__(self) -> str:
        return f"SubSample{self.params[0].shape, self.params[0].shape}"


# Radial Basis Function layer
class RBF(object):
    
    def __init__(self, outputs) -> None:
        super().__init__()
        self.outputs = outputs
        self.cache = None
        self.params = None, None
        self.in_shape, self.out_shape = None, None
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (1, 1)
        self.param_shape = (self.outputs, np.product(self.in_shape))
        # self.params = ( np.random.choice([-1, 1], self.param_shape) , )
        self.params = (rbf_initialize(), ) # initializes output basis vectors for each class
        assert self.param_shape == self.params[0].shape
        return self
    
    def __call__(self, input, label, mode = 'test'):
        if mode == 'test': return self.predict(input)
        self.cache = input , self.params[0][label, :]
        # calculating (xj - wij) ** 2, sum squared from label basis vector and input array
        return np.sum(0.5 * np.sum((self.cache[0] - self.cache[1]) ** 2, axis = 1, keepdims=True))
    
    def predict(self, input):
        # returns the index of the basic vector with minimum sub squared distance from input array from dense layer
        sq_diff = np.square(input[:, np.newaxis, :] - np.array([self.params[0]] * input.shape[0]))
        sq_diff_sum = np.sum(sq_diff, axis=2)
        pred = np.argmin(sq_diff_sum, axis = 1)
        return pred

    def __gradients__(self, next_d = 1):
        return None, None, next_d * (self.cache[0] - self.cache[1]) # straight forward derivative of sum squared distance wrt input tensor
    
    def __str__(self) -> str:
        return f"RBF{self.param_shape}"

# Fully Connected Layer
class Dense(BaseLayer):

    def __init__(self, outputs) -> None:
        super().__init__()
        self.outputs = outputs
        self.cache = None
        self.params = None, None
        self.in_shape, self.out_shape = None, None
    
    def __call__(self, input):
        self.cache = (input.reshape((input.shape[0]), np.prod(list(input.shape)[1:])), input.shape)
        return  np.matmul(self.cache[0] , self.params[0]) + self.params[1]
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (self.outputs, 1)
        self.param_shape = (np.product(self.in_shape), self.out_shape[0])
        self.params = initialize(self.param_shape)
        return self

    def __gradients__(self, next_d):
        # derivatives wrt W, b and input tensor ( uses basic  I @ W  + b formulation)
        return np.matmul(self.cache[0].T ,  next_d), np.sum(next_d.T, axis = 1), np.matmul(next_d, self.params[0].T).reshape(self.cache[1])
    
    def __str__(self) -> str:
        return f"Dense{self.param_shape, self.params[1].shape}"
    
    
    
""" *************** Activation *************** """

# activation class, added after any layer if non-linearity is to be introduced
class Activation(object):

#     initialisation for activation function, default is tanh
    def __init__(self, name = 'tanh') -> None:
        super().__init__()
        self.name = name
        self.cache = None
        self.params = None
        self.in_shape, self.out_shape = None, None
        self.func, self.func_d = __builtin_funcs__[name]

    def __call__(self, input):
        # apply the registered activation
        output, self.cache = self.func(input), input
        return output
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = in_shape
        return self

    def __gradients__(self, next_d):
        # computing registered activation derivative of input
        return None, None, next_d * self.func_d(self.cache)
    
    def __str__(self) -> str:
        return f"Activation({self.name})"