import numpy as np
import torch


def one_hot_encode(labels, num_classes):
    # 创建一个全零矩阵
    one_hot_matrix = np.zeros((len(labels), num_classes))
    # 将对应位置设置为 1
    for i, label in enumerate(labels):
        one_hot_matrix[i, label] = 1
    return one_hot_matrix

# 示例
labels = [0, 2, 1]  # 假设类别为 0, 1, 2
num_classes = 3
one_hot_encoded = one_hot_encode(labels, num_classes)
print(one_hot_encoded)


np.random.seed(1)
torch.manual_seed(1)

a = [1, 0 , 0]
a = torch.FloatTensor([a])
b = torch.randn(3,4)
print(b)
print(torch.matmul(a,b))
