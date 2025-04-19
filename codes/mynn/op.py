from abc import abstractmethod
import cupy as cp
from cupy.lib.stride_tricks import as_strided


class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Linear(Layer):
    def __init__(self, in_dim, out_dim, initialize_method=cp.random.normal,
                 weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return cp.dot(X, self.W) + self.b

    def backward(self, grad):
        self.grads['W'] = cp.dot(self.input.T, grad)
        self.grads['b'] = cp.sum(grad, axis=0)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        return cp.dot(grad, self.W.T)

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if initialize_method is None:
            std = cp.sqrt(2. / (in_channels * kernel_size ** 2))
            initialize_method = lambda size: cp.random.normal(0, std, size)
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.X_padded = None
        self.patches_reshaped = None

    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)

    def forward(self, X):
        batch_size, in_channels, H, W = X.shape
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.X_padded = cp.pad(X, ((0, 0), (0, 0),
                                   (self.padding, self.padding),
                                   (self.padding, self.padding)), mode='constant')

        batch_stride, channel_stride, h_stride, w_stride = self.X_padded.strides
        patches = as_strided(
            self.X_padded,
            shape=(batch_size, self.in_channels, H_out, W_out,
                   self.kernel_size, self.kernel_size),
            strides=(batch_stride, channel_stride,
                     h_stride * self.stride,
                     w_stride * self.stride,
                     h_stride, w_stride)
        )

        self.patches_reshaped = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1,
                                                                            self.in_channels * self.kernel_size ** 2)
        W_reshaped = self.W.reshape(self.out_channels, -1)
        output = cp.dot(self.patches_reshaped, W_reshaped.T).reshape(batch_size,
                                                                     H_out, W_out, self.out_channels)
        return output.transpose(0, 3, 1, 2) + self.b

    def backward(self, grads):
        batch_size, out_channels, H_out, W_out = grads.shape
        in_channels = self.in_channels
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        dout_reshaped = grads.transpose(0, 2, 3, 1).reshape(-1, out_channels)
        dW = cp.dot(dout_reshaped.T, self.patches_reshaped).reshape(self.W.shape)
        db = cp.sum(grads, axis=(0, 2, 3)).reshape(1, out_channels, 1, 1)

        W_reshaped = self.W.reshape(out_channels, -1)
        col_grad = cp.dot(dout_reshaped, W_reshaped)
        col_grad_reshaped = col_grad.reshape(batch_size, H_out, W_out, in_channels, kernel_size, kernel_size)

        dX_padded = cp.zeros_like(self.X_padded)
        batch_stride, channel_stride, h_stride, w_stride = dX_padded.strides
        patches_dX = as_strided(
            dX_padded,
            shape=(batch_size, in_channels, H_out, W_out, kernel_size, kernel_size),
            strides=(
                batch_stride,
                channel_stride,
                h_stride * stride,
                w_stride * stride,
                h_stride,
                w_stride
            )
        )

        col_grad_reshaped = col_grad_reshaped.transpose(0, 3, 1, 2, 4, 5)
        patches_dX += col_grad_reshaped

        if padding == 0:
            dX = dX_padded
        else:
            dX = dX_padded[:, :, padding:-padding, padding:-padding]

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db
        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}
        self.X_padded = None
        self.patches_reshaped = None


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        return cp.where(X < 0, 0, X)

    def backward(self, grads):
        return cp.where(self.input < 0, 0, grads)


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.optimizable = False

    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)

    def forward(self, X):
        N, C, H, W = X.shape
        self.X = X
        H_out = H // self.kernel_size
        W_out = W // self.kernel_size
        X_reshaped = X.reshape(N, C, H_out, self.kernel_size, W_out, self.kernel_size)
        return X_reshaped.max(axis=(3, 5))

    def backward(self, grad):
        N, C, H_out, W_out = grad.shape
        grad_reshaped = grad.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3)
        X_reshaped = self.X.reshape(N, C, H_out, self.kernel_size, W_out, self.kernel_size)
        mask = (X_reshaped == X_reshaped.max(axis=(3, 5), keepdims=True))
        return grad_reshaped * mask.reshape(N, C, H_out * self.kernel_size, W_out * self.kernel_size)


class MultiCrossEntropyLoss(Layer):
    def __init__(self, model=None, max_classes=10):
        self.model = model
        self.has_softmax = True

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        max_logit = cp.max(predicts, axis=1, keepdims=True)
        shifted_logits = predicts - max_logit
        exp_shifted = cp.exp(shifted_logits)
        sum_exp = cp.sum(exp_shifted, axis=1, keepdims=True)
        log_probs = shifted_logits - cp.log(sum_exp)
        true_log_probs = log_probs[cp.arange(len(labels)), labels]
        loss = -cp.mean(true_log_probs)
        return loss

    def backward(self):
        batch_size = self.predicts.shape[0]
        max_logit = cp.max(self.predicts, axis=1, keepdims=True)
        shifted_logits = self.predicts - max_logit
        exp_shifted = cp.exp(shifted_logits)
        sum_exp = cp.sum(exp_shifted, axis=1, keepdims=True)
        softmax_output = exp_shifted / sum_exp
        y_true = cp.zeros_like(softmax_output)
        y_true[cp.arange(batch_size), self.labels] = 1
        self.grads = (softmax_output - y_true) / batch_size
        self.model.backward(self.grads)
        return self.grads


class L2Regularization(Layer):
    def __init__(self, model, weight_decay_lambda=1e-4):
        super().__init__()
        self.model = model
        self.weight_decay_lambda = weight_decay_lambda

    def forward(self, X):
        l2_loss = 0.0
        for layer in self.model.layers:
            if hasattr(layer, 'W'):
                l2_loss += 0.5 * self.weight_decay_lambda * cp.sum(layer.W ** 2)
        return l2_loss

    def backward(self):
        for layer in self.model.layers:
            if hasattr(layer, 'W'):
                layer.grads['W'] += self.weight_decay_lambda * layer.W


def softmax(X):
    x_max = cp.max(X, axis=1, keepdims=True)
    x_exp = cp.exp(X - x_max)
    partition = cp.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition