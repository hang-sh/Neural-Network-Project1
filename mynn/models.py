from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    # def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_prob=None):
    def __init__(self, d_in=None, nHidden=[], d_out=None, act_func=None, lambda_list=None, dropout_prob=None):
        # self.size_list = size_list
        assert isinstance(nHidden, list), "nHidden must be a list."

        self.size_list = [d_in] + nHidden + [d_out]
        self.act_func = act_func

        # 新增，判断是否使用L2正则化
        self.l2_reg = False
        if lambda_list is not None:
            self.l2_reg = True
        # 新增，dropout
        self.dropout_prob = dropout_prob

        if self.size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
                    if dropout_prob is not None and i == 0:
                        layer_d = Dropout(prob=1-dropout_prob)
                        self.layers.append(layer_d)

    def __call__(self, X, is_train=True):
        return self.forward(X, is_train)

    def forward(self, X, is_train=True):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            if isinstance(layer, Dropout):
                outputs = layer(outputs, is_train)
            else:
                outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.dropout_prob = param_list[2]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 3]['W']
                layer.b = param_list[i + 3]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 3]['weight_decay']
                layer.weight_decay_lambda = param_list[i+3]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
                    if self.dropout_prob is not None and i == 0:
                        layer_d = Dropout(prob=1-self.dropout_prob)
                        self.layers.append(layer_d)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func, self.dropout_prob]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN_1(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, channels_list=None, kernel_sizes=None, pool_kernel_sizes=None, size_list=None, act_func=None, lambda_list=None, dropout_prob=None):
        self.channels_list = channels_list
        self.kernel_sizes = kernel_sizes
        self.pool_kernel_sizes = pool_kernel_sizes
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_prob = dropout_prob

        # 新增，判断是否使用L2正则化
        self.l2_reg = False
        if lambda_list is not None:
            self.l2_reg = True
        
        if channels_list is not None and kernel_sizes is not None:
            assert len(channels_list) == len(kernel_sizes) + 1, \
                "Length of channels_list should be one more than length of kernel_sizes."
            self.layers = []
            for i in range(len(channels_list) - 1):
                # 卷积层
                layer = conv2D(channels_list[i], channels_list[i+1], kernel_sizes[i], stride=1, padding=1)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)
                # ReLU 层
                layer_f = ReLU()
                self.layers.append(layer_f)
                # Maxpooling 层
                if self.pool_kernel_sizes is not None and pool_kernel_sizes[i] is not None:
                    layer_p = MaxPooling(kernel_size=pool_kernel_sizes[i])
                    self.layers.append(layer_p)
        
        # 展平层
        self.layers.append(Flatten())

        # 全连接层
        if size_list is not None and act_func is not None:
            for i in range(len(size_list)-1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[len(kernel_sizes)+i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    if dropout_prob is not None:
                        layer_d = Dropout(prob=1-dropout_prob)
                        self.layers.append(layer_d)

        # pass

    def __call__(self, X, is_train=True):
        # print(f"X: {X.shape}, is_train: {is_train}")
        return self.forward(X, is_train)

    def forward(self, X, is_train=True):
        assert self.channels_list is not None and self.kernel_sizes is not None, \
            'Model has not initialized yet. Use model.load_model to load a model or create a new model with channels_list and kernel_sizes offered.'
        assert self.size_list is not None and self.act_func is not None,\
            'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'

        outputs = X
        # print(f"outputs: {outputs.shape}")
        for layer in self.layers:
            if isinstance(layer, Dropout):
                outputs = layer(outputs, is_train)
            else:
                # print(f"outputs: {outputs.shape}")
                # print(layer.__class__.__name__)
                outputs = layer(outputs)
                # print(f"outputs: {outputs.shape}")
        return outputs
        # pass

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

        # pass
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        
        self.channels_list = param_list[0]
        self.kernel_sizes = param_list[1]
        self.pool_kernel_sizes = param_list[2]
        self.size_list = param_list[3]
        self.act_func = param_list[4]
        self.dropout_prob = param_list[5]
        s1 = 6
        self.layers = []
        for i in range(len(self.channels_list) - 1):
            layer = conv2D(self.channels_list[i], self.channels_list[i+1],self.kernel_sizes[i], stride=1, padding=1)
            layer.W = param_list[i + s1]['W']
            layer.b = param_list[i + s1]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + s1]['weight_decay']
            layer.weight_decay_lambda = param_list[i + s1]['lambda']
            self.layers.append(layer)
            # ReLU 层
            layer_f = ReLU()
            self.layers.append(layer_f)
            # Maxpooling 层
            if self.pool_kernel_sizes is not None:
                layer_p = MaxPooling(kernel_size=self.pool_kernel_sizes[i], stride=self.pool_kernel_sizes[i])
                self.layers.append(layer_p)
            
        
        self.layers.append(Flatten())
        s2 = len(self.channels_list) + s1
        
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i+s2]['W']
            layer.b = param_list[i+s2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i+s2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+s2]['lambda']
            
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
                if self.dropout_prob is not None:
                    layer_d = Dropout(prob=1-self.dropout_prob)
                    self.layers.append(layer_d)

                
        # pass
        
    def save_model(self, save_path):
        param_list = [self.channels_list, self.kernel_sizes, self.pool_kernel_sizes, self.size_list, self.act_func, self.dropout_prob]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        
        # pass


class Model_CNN:
    """    
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, channels_list=None, kernel_sizes=None, pool_kernel_sizes=None, size_list=None, act_func='ReLU', lambda_list=None, dropout_prob=None):
        self.channels_list = channels_list
        self.kernel_sizes = kernel_sizes
        self.pool_kernel_sizes = pool_kernel_sizes
        self.size_list = size_list
        self.act_func = act_func
        self.dropout_prob = dropout_prob
        self.l2_reg = lambda_list is not None
        
        self.layers = []
        if channels_list and kernel_sizes:
            self.build_conv_layers(channels_list, kernel_sizes, lambda_list)
        # 展平层
        self.layers.append(Flatten())
        if size_list and act_func:
            self.build_fc_layers(size_list, act_func, lambda_list)

    def build_conv_layers(self, channels_list, kernel_sizes, lambda_list):
        for i in range(len(channels_list) - 1):
            layer = conv2D(channels_list[i], channels_list[i+1], kernel_sizes[i], stride=1, padding=1)
            if lambda_list:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[i]
            self.layers.append(layer)
            
            # 添加BatchNormalization层
            self.layers.append(BatchNormalization())
            
            self.layers.append(ReLU())  # 默认使用ReLU激活函数
            if self.pool_kernel_sizes and self.pool_kernel_sizes[i]:
                self.layers.append(MaxPooling(kernel_size=self.pool_kernel_sizes[i]))

    def build_fc_layers(self, size_list, act_func, lambda_list):
        for i in range(len(size_list)-1):
            layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
            if lambda_list:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[len(self.kernel_sizes)+i]
            self.layers.append(layer)
            if i < len(size_list) - 2:
                if act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif act_func == 'Sigmoid':
                    raise NotImplementedError
                if self.dropout_prob is not None:
                    self.layers.append(Dropout(prob=1-self.dropout_prob))

    def __call__(self, X, is_train=True):
        return self.forward(X, is_train)

    def forward(self, X, is_train=True):
        outputs = X
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                outputs = layer(outputs, is_train)
            else:
                outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grads = layer.backward(grads)
        return grads

    def load_model(self, param_list_path):
        with open(param_list_path, 'rb') as f:
            param_list = pickle.load(f)
        
        self.channels_list, self.kernel_sizes, self.pool_kernel_sizes, self.size_list, self.act_func, self.dropout_prob = param_list[:6]
        self.layers = []
        self.build_conv_layers(self.channels_list, self.kernel_sizes, None)
        self.layers.append(Flatten())
        self.build_fc_layers(self.size_list, self.act_func, None)
        
        s1 = 6
        i = 0
        for _, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                layer.params['W'] = param_list[s1+i]['W']
                layer.params['b'] = param_list[s1+i]['b']
                if hasattr(layer, 'gamma'):
                    layer.gamma = param_list[s1+i]['gamma']
                    layer.beta = param_list[s1+i]['beta']
                layer.weight_decay = param_list[s1+i]['weight_decay']
                layer.weight_decay_lambda = param_list[s1+i]['lambda']
                i += 1

    def save_model(self, save_path):
        param_list = [self.channels_list, self.kernel_sizes, self.pool_kernel_sizes, self.size_list, self.act_func, self.dropout_prob]
        for layer in self.layers:
            if hasattr(layer, 'params'):
                params = {'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay, 'lambda': layer.weight_decay_lambda}
                if hasattr(layer, 'gamma'):
                    params.update({'gamma': layer.gamma, 'beta': layer.beta})
                param_list.append(params)
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)