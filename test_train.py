# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

from mynn.utils import EarlyStopping
from mynn.utils import augment_data

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# Data augmentation
# train_imgs = train_imgs.reshape(-1, 1, 28, 28)
# augmented_train_imgs, augmented_train_labs = augment_data(train_imgs, train_labs, augment_factor=1)
# train_imgs = train_imgs.reshape(-1, 784)

# augmented_train_imgs = augmented_train_imgs.reshape(-1, 28*28)
# augmented_train_labs = np.squeeze(augmented_train_labs)  

# Normalize 
# min_val = augmented_train_imgs.min()
# max_val = augmented_train_imgs.max()
# augmented_train_imgs = (augmented_train_imgs - min_val) / (max_val - min_val)
# mean = augmented_train_imgs.mean()
# std = augmented_train_imgs.std()
# augmented_train_imgs = (augmented_train_imgs - mean) / std

# Combination
# combined_images = np.concatenate([train_imgs, augmented_train_imgs], axis=0)
# combined_labels = np.concatenate([train_labs, augmented_train_labs], axis=0)


nHidden = [600, 100]
lambda_list = [1e-4] * (len(nHidden) + 1)  # L2 regularization 
linear_model = nn.models.Model_MLP(train_imgs.shape[-1], nHidden, 10, 'ReLU')
optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
early_stopping = EarlyStopping(patience=1, min_delta=0.01)
runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=3, log_iters=100, save_dir=r'./best_models')
# runner.train([combined_images, combined_labels], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
plt.show()

# CNN Model
# Reshape the images for CNN (batch_size, channels, height, width)
# train_imgs = train_imgs.reshape(-1, 1, 28, 28)
# valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

# valid_imgs_1 = valid_imgs[:1000,]
# valid_labs_1 = valid_labs[:1000,]

# cnn_model = nn.models.Model_CNN(
#        channels_list=[1, 16, 32, 64],
#        kernel_sizes=[3, 3, 3],
#        pool_kernel_sizes=[2, 2, 2],
#        size_list=[64*3*3, 128, 10],
#        act_func='ReLU',
#        lambda_list=None,  # [1e-4]*5,
#        dropout_prob=None
#)
# optimizer = nn.optimizer.SGD(init_lr=0.1, model=cnn_model)#, clip_value=2.0)
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[500, 800, 1000], gamma=0.5)
# loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)
# early_stopping = EarlyStopping(patience=1, min_delta=0.01)
# runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, early_stopping=early_stopping)
# runner.train([train_imgs, train_labs], [valid_imgs_1, valid_labs_1], num_epochs=2, log_iters=100, save_dir=r'./best_models')

# _, axes = plt.subplots(1, 2)
# axes.reshape(-1)
# _.set_tight_layout(1)
# plot(runner, axes)
# plt.show()