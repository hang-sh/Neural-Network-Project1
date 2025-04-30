from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size = (in_dim, out_dim))
        self.b = initialize_method(size= (1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.matmul(X, self.params['W']) + self.params['b']


    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        m = grad.shape[0]
        
        self.grads['W'] = np.matmul(self.input.T, grad) / m 
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / m

        return np.matmul(grad, self.params['W'].T) 
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


# He initialization 
def he_initialization(kernel_shape):
    """
    使用 He 初始化方法生成卷积核权重。
    :param kernel_shape: 卷积核的形状，格式为 (out_channels, in_channels, kernel_height, kernel_width)
    :return: 初始化后的卷积核权重
    """
    out_channels, in_channels, kernel_height, kernel_width = kernel_shape
    n_in = in_channels * kernel_height * kernel_width
    std = np.sqrt(2.0 / n_in)
    weights = np.random.randn(*kernel_shape) * std
    
    return weights

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=he_initialization, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        self.W = initialize_method((out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels,))
        self.params = {'W' : self.W, 'b' : self.b}

        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.
        # pass

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        batch_size, _ , H, W = X.shape
        self.input = X
        H_out = (H + 2 * self.padding - self.kernel_size)//self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size)//self.stride + 1
        
        if self.padding > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        else:
            X_pad = X

        output = np.zeros((batch_size, self.out_channels, H_out, W_out))
        
        for idx in range(batch_size):
            for k in range(self.out_channels):
                for j in range(H_out):
                    for i in range(W_out):
                        w_start = i * self.stride
                        w_end = w_start + self.kernel_size
                        h_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        X_patch = X_pad[idx, :, h_start:h_end, w_start:w_end] 
                        output[idx][k][j][i] = np.sum(X_patch * self.params['W'][k]) + self.params['b'][k]                           
        return output
        # pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _ , H_out, W_out = grads.shape

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.input)

        if self.padding > 0:
            dX_pad = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        else:
            dX_pad = self.input

        for idx in range(batch_size):
            for k in range(self.out_channels):
                for j in range(H_out):
                    for i in range(W_out):
                        w_start = i * self.stride
                        w_end = w_start + self.kernel_size
                        h_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        
                        X_patch = dX_pad[idx, :, h_start:h_end, w_start:w_end]
                        dW[k] += grads[idx][k][j][i] * X_patch
                        db[k] += grads[idx][k][j][i]
                        dX_pad[idx, :, h_start:h_end, w_start:w_end] += grads[idx][k][j][i] * self.params['W'][k]
        
        if self.padding > 0:
            dX = dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_pad
            
        self.grads['W'] = dW / batch_size
        self.grads['b'] = db / batch_size  
        return dX     
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class BatchNormalization(Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.optimizable = False
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None
        self.cache = None
        self.initialized = False  # 标记是否已完成初始化

    def _initialize_parameters(self, input_shape):
        """根据输入形状初始化参数"""
        n_out = input_shape[1]
        self.gamma = np.ones((n_out,))
        self.beta = np.zeros((n_out,))
        self.running_mean = np.zeros((n_out,))
        self.running_var = np.ones((n_out,))
        self.initialized = True

    def __call__(self, X, is_train=True):
        return self.forward(X, is_train)
    
    def forward(self, X, is_train=True):
        if not self.initialized:
            self._initialize_parameters(X.shape)

        if is_train:
            mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
            var = np.var(X, axis=(0, 2, 3), keepdims=True)
            X_norm = (X - mean) / np.sqrt(var + self.epsilon)
            out = self.gamma[None, :, None, None] * X_norm + self.beta[None, :, None, None]

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * np.squeeze(mean, axis=(0, 2, 3))
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * np.squeeze(var, axis=(0, 2, 3))
            self.cache = (X, X_norm, mean, var)
        else:
            X_norm = (X - self.running_mean[None, :, None, None]) / np.sqrt(self.running_var[None, :, None, None] + self.epsilon)
            out = self.gamma[None, :, None, None] * X_norm + self.beta[None, :, None, None]

        return out

    def backward(self, dout):
        X, X_norm, mean, var = self.cache
        m = X.shape[0] * X.shape[2] * X.shape[3]

        dX_norm = dout * self.gamma[None, :, None, None]
        dvar = np.sum(dX_norm * (X - mean) * (-0.5) * (var + self.epsilon) ** (-1.5), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dX_norm * (-1 / np.sqrt(var + self.epsilon)), axis=(0, 2, 3), keepdims=True)
        dmean += dvar * np.mean(-2 * (X - mean), axis=(0, 2, 3), keepdims=True)

        dX = dX_norm / np.sqrt(var + self.epsilon) + (dvar * 2 * (X - mean)) / m + dmean / m

        dgamma = np.sum(dout * X_norm, axis=(0, 2, 3))
        dbeta = np.sum(dout, axis=(0, 2, 3))
        self.grads = {'gamma': dgamma, 'beta': dbeta}
        return dX


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.has_softmax = True
        self.model = model
        self.grads = None
        self.predicts = None
        self.labels = None
        self.max_classes = max_classes

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.predicts = predicts
        self.labels = labels

        if self.has_softmax:
            self.predicts = softmax(self.predicts)

        y_true = np.zeros(self.predicts.shape)
        y_true[np.arange(self.predicts.shape[0]), self.labels] = 1
        loss = -np.sum(y_true * np.log(self.predicts + 1e-15)) / y_true.shape[0]
        return loss

    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        y_true = np.zeros(self.predicts.shape)
        y_true[np.arange(self.predicts.shape[0]), self.labels] = 1
        if self.has_softmax:
            self.grads = (self.predicts - y_true) / self.predicts.shape[0]
        else:
            self.grads = - (y_true / self.predicts) / self.predicts.shape[0]

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model=None) -> None:
        self.model = model

    def forward(self):
        l2_loss = 0
        for layer in self.model.layers:
            if hasattr(layer, "params") and 'W' in layer.params:
                if layer.weight_decay:
                    l2_loss += layer.weight_decay_lambda * np.sum(layer.params['W'] ** 2)
        return 0.5 * l2_loss
    
    def backward(self):
        for layer in self.model.layers:
            if hasattr(layer, "params") and 'W' in layer.params:
                if layer.weight_decay:
                    layer.grads['W'] += layer.weight_decay_lambda * layer.params['W']

    # pass

class Dropout(Layer):
    """
    A dropout layer.
    """
    def __init__(self, prob = 0.5):
        """
        prob: the probability of keeping a neuron active.
        """
        super().__init__()
        self.optimizable =False
        self.prob = prob
        self.mask = None

    def __call__(self, X, is_train=True):
        return self.forward(X, is_train)

    def forward(self, X, is_train=True):
        if is_train:
            self.mask = np.random.binomial(1, self.prob, size=X.shape) / self.prob
            return X * self.mask
        else:
            return X * self.prob
    
    def backward(self, grads):
        return grads * self.mask


class MaxPooling(Layer):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.optimizable =False
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        """
        batch_size, channels , H, W = X.shape
        self.input = X
        H_out = (H - self.kernel_size)//self.stride + 1
        W_out = (W - self.kernel_size)//self.stride + 1

        output = np.zeros((batch_size, channels, H_out, W_out))
        
        for idx in range(batch_size):
            for k in range(channels):
                for j in range(H_out):
                    for i in range(W_out):
                        w_start = i * self.stride
                        w_end = w_start + self.kernel_size
                        h_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        
                        X_patch = X[idx, k, h_start:h_end, w_start:w_end]
                        output[idx][k][j][i] = np.max(X_patch)

        return output
        
    
    def backward(self, grads):
        """
        grads : [batch_size, channels, new_H, new_W]
        """
        batch_size, channels, H_out, W_out = grads.shape
        dX = np.zeros_like(self.input)

        for idx in range(batch_size):
            for k in range(channels):
                for j in range(H_out):
                    for i in range(W_out):
                        w_start = i * self.stride
                        w_end = w_start + self.kernel_size
                        h_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        
                        X_patch = self.input[idx, k, h_start:h_end, w_start:w_end]
                        mask = (X_patch == np.max(X_patch))
                        dX[idx, k, h_start:h_end, w_start:w_end] += mask * grads[idx, k, j, i]
        return dX
        
    

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable =False
        self.shape = None

    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grads):
        return grads.reshape(self.shape)
        
        
# Softmax function      
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

