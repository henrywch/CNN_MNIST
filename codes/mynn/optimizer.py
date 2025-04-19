from abc import abstractmethod
import cupy as cp


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocities = {}
        for layer in model.layers:
            if layer.optimizable:
                self.velocities[id(layer)] = {key: cp.zeros_like(param) for key, param in layer.params.items()}

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params:
                    v = self.velocities[id(layer)][key]
                    v = self.mu * v - self.init_lr * layer.grads[key]
                    layer.params[key] += v
                    self.velocities[id(layer)][key] = v


class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

        for layer in model.layers:
            if layer.optimizable:
                self.m[id(layer)] = {key: cp.zeros_like(param) for key, param in layer.params.items()}
                self.v[id(layer)] = {key: cp.zeros_like(param) for key, param in layer.params.items()}

    def step(self):
        self.t += 1
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params:
                    g = layer.grads[key]
                    self.m[id(layer)][key] = self.beta1 * self.m[id(layer)][key] + (1 - self.beta1) * g
                    self.v[id(layer)][key] = self.beta2 * self.v[id(layer)][key] + (1 - self.beta2) * g ** 2
                    m_hat = self.m[id(layer)][key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[id(layer)][key] / (1 - self.beta2 ** self.t)
                    layer.params[key] -= self.init_lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)