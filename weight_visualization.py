# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

def visualize_top_W1(W1, img_shape=(28, 28, 1), top_k=16, filename='W1_top_visual.png'):

    save_path = './figs/' + filename

    norms = np.linalg.norm(W1, axis=0)
    top_indices = np.argsort(norms)[-top_k:][::-1]  # 从大到小排列

    num_cols = int(np.sqrt(top_k))
    num_rows = (top_k + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))

    for i, idx in enumerate(top_indices):
        ax = axes[i // num_cols, i % num_cols]
        weight = W1[:, idx]
        img = weight.reshape(img_shape)

        # 转为灰度图
        img_gray = img.mean(axis=2)
        img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())

        ax.matshow(img_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
        ax.axis('off')

    plt.suptitle(f"W1 中权重范数最大的前 {top_k} 个神经元", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.show()

def visualize_weight_heatmap(W, title="权重热力图", filename="heatmap.png"):
    save_path = './figs/' + filename

    plt.figure(figsize=(10, 6))
    plt.imshow(W, aspect='auto', cmap='seismic')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("下一层神经元")
    plt.ylabel("上一层神经元")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_weight_hist(weights, title="权重分布直方图", bins=50, filename="weight_hist.png"):
    save_path = './figs/' + filename

    plt.figure(figsize=(6, 4))
    plt.hist(weights.flatten(), bins=bins, color='steelblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("权重值")
    plt.ylabel("频数")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_conv_filters(conv_layer):
    weights = conv_layer.params['W']
    num_filters, num_channels, height, width = weights.shape
    
    # 合并所有通道的绝对值之和作为单个图像
    combined_weights = np.abs(weights).sum(axis=1)  # 形状变为 (num_filters, height, width)
    
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    
    for i in range(num_filters):
        filter_img = combined_weights[i]
        ax = axes.flat[i]
        ax.imshow(filter_img, cmap='gray', vmin=combined_weights.min(), vmax=combined_weights.max())
        ax.axis('off')
    
    for i in range(num_filters, grid_size**2):
        fig.delaxes(axes.flat[i])
    plt.show()

# model = nn.models.Model_MLP()
model = nn.models.Model_CNN()
model.load_model(r'.\saved_models\best_model.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

mats = []
mats.append(model.layers[0].params['W'])
mats.append(model.layers[2].params['W'])

#_, axes = plt.subplots(30, 20)
#_.set_tight_layout(1)
#axes = axes.reshape(-1)
#for i in range(600):
    #axes[i].matshow(mats[0].T[i].reshape(28,28))
    #axes[i].set_xticks([])
    #axes[i].set_yticks([])

# plt.figure()
plt.matshow(mats[1])
plt.xticks([])
plt.yticks([])
plt.show()

visualize_top_W1(model.layers[0].params['W'])
visualize_weight_heatmap(model.layers[2].params['W'], title="W2 权重热力图", filename="W2_heatmap.png")
plot_weight_hist(model.layers[2].params['W'], title="W2 权重分布直方图", filename="W2_hist.png")

# visualize_conv_filters(model.layers[0])
