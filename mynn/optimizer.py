from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model, clip_value=None):
        super().__init__(init_lr, model)
        self.clip_value = clip_value
    
    def step(self):
        if self.clip_value is not None:
            clip_gradients_by_value(self.model, self.clip_value)
            
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, beta=0.9):
        super().__init__(init_lr, model)
        self.beta = beta
        self.prev_params = {} # 存储上一次的参数
        for layer in self.model.layers:
            if layer.optimizable == True:
                self.prev_params[layer] = {}
                for key in layer.params.keys():
                    self.prev_params[layer][key] = np.zeros_like(layer.params[key])
        # pass
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    momentum = self.beta * (layer.params[key] - self.prev_params[layer][key]) 
                    self.prev_params[layer][key] = layer.params[key].copy()  # 更新上一次的参数
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key] + momentum
        # pass


def clip_gradients_by_value(model, clip_value):
    """
    Clip gradients by value.
    Args:
        model: The model containing parameters and gradients.
        clip_value: The maximum allowed absolute value for gradients.
    """
    for layer in model.layers:
        if hasattr(layer, 'grads'):  # 检查是否有梯度
            for key in layer.grads:
                # 使用 np.clip 将梯度裁剪到 [-clip_value, clip_value] 范围内
                np.clip(layer.grads[key], -clip_value, clip_value, out=layer.grads[key])

