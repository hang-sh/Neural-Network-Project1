import os
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import zoom

class EarlyStopping:

    def __init__(self, patience=1, min_delta=0.01, save_dir=r"./best_models"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.best_score = 0
        self.counter = 0

    def check(self, dev_score, model):
        if self.best_score == 0 or dev_score > self.best_score + self.min_delta:
            self.counter = 0
            save_path = os.path.join(self.save_dir, 'best_model.pickle')
            self.save_model(model, save_path)
            print(f"best accuracy performence has been updated: {self.best_score:.5f} --> {dev_score:.5f}")
            self.best_score = dev_score

        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                return True
        return False
    
    def save_model(self, model, save_path):
        model.save_model(save_path)

# Data Augmentation
def random_translation(image, max_shift=2):
    """
    随机平移图像。
    :param image: 输入图像，形状为 (1, 28, 28)
    :param max_shift: 最大平移像素数
    :return: 平移后的图像，形状为 (1, 28, 28)
    """
    _, h, w = image.shape
    dx = np.random.randint(-max_shift, max_shift + 1)  # 水平平移
    dy = np.random.randint(-max_shift, max_shift + 1)  # 垂直平移
    
    # 创建空白矩阵
    translated_image = np.zeros_like(image[0])
    
    # 确保切片操作不会越界
    source_y_start = max(0, -dy)
    source_y_end = min(h, h - dy)
    source_x_start = max(0, -dx)
    source_x_end = min(w, w - dx)

    target_y_start = max(0, dy)
    target_y_end = min(h, h + dy)
    target_x_start = max(0, dx)
    target_x_end = min(w, w + dx)
    
    # 执行平移操作
    translated_image[target_y_start:target_y_end, target_x_start:target_x_end] = \
        image[0][source_y_start:source_y_end, source_x_start:source_x_end]
    
    return translated_image.reshape(1, h, w)  # 恢复通道维度
def random_rotation(image, max_angle=5):
    """
    随机旋转图像。
    image: 输入图像，形状为 (1, 28, 28)
    max_angle: 最大旋转角度
    """
    angle = np.random.uniform(-max_angle, max_angle)
    rotated_image = rotate(image[0], angle, reshape=False, mode='nearest') # 使用最近邻插值
    return rotated_image.reshape(1, *rotated_image.shape)

def random_scaling(image, scale_range=(0.95, 1.05)):
    """
    随机缩放图像。
    image: 输入图像，形状为 (1, 28, 28)
    scale_range: 缩放比例范围
    """
    _, h, w = image.shape
    resized_image = np.zeros_like(image[0])
    scale = np.random.uniform(scale_range[0], scale_range[1])
    zoomed_image = zoom(image[0], scale, order=1)  # 使用线性插值进行缩放
    z_h, z_w = zoomed_image.shape
    if scale > 1:
        start_x = (z_h - h) // 2
        start_y = (z_w - w) // 2
        resized_image = zoomed_image[start_x:start_x + h, start_y:start_y + w]
    else:
        start_x = (h - z_h) // 2
        start_y = (w - z_w) // 2
        resized_image[start_x:start_x + z_h, start_y:start_y + z_w] = zoomed_image
    
    return resized_image.reshape(1, h, w)  # 恢复通道维度  

def augment_data(images, labels, augment_factor=3):
    """
    对数据集进行增强。
    images: 输入图像，形状为 (N, 1, 28, 28)
    labels: 对应标签，形状为 (N,)
    augment_factor: 每个样本生成的增强样本数量
    """
    augmented_images = []
    augmented_labels = []
    for i in range(len(images)):
        for _ in range(augment_factor):
            # 随机选择增强方法
            method = np.random.choice(['translation', 'rotation', 'scaling'])

            if method == 'translation':
                augmented_image = random_translation(images[i])
            elif method == 'rotation':
                augmented_image = random_rotation(images[i])
            elif method == 'scaling':
                augmented_image = random_scaling(images[i])
            
            augmented_images.append(augmented_image)
            augmented_labels.append(labels[i])
    
    return np.array(augmented_images), np.array(augmented_labels)