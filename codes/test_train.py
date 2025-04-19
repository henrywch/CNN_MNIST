# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
import cupy as cp
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


idx = np.random.permutation(np.arange(num))
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

train_imgs = cp.asarray(train_imgs)
train_labs = cp.asarray(train_labs)
valid_imgs = cp.asarray(valid_imgs)
valid_labs = cp.asarray(valid_labs)

cnn_model = nn.models.Model_CNN()
base_lr = 0.001
batch_size = 32
optimizer = nn.optimizer.Adam(init_lr=base_lr * (batch_size / 32), model=cnn_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 1600, 2400], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=10, log_iters=100, save_dir=r'./best_models')


_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()