import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_directionality(point_cloud):
    # 假设point_cloud是一个N x 3的数组，每行代表一个点的坐标

    directionality = []

    for point in point_cloud:
        neighbors = get_neighbors(point, point_cloud)

        if len(neighbors) < 3:
            directionality.append(0)  # 如果邻居数量小于3，则方向性度为0
        else:
            neighbors = np.array(neighbors)  # 转换为NumPy数组
            covariance_matrix = np.cov(neighbors.T)  # 计算3x3协方差矩阵
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            # 选择最小特征值对应的特征向量（即最小主轴）
            min_eigenvalue_index = np.argmin(eigenvalues)
            min_eigenvector = eigenvectors[:, min_eigenvalue_index]

            # 计算方向性度
            directionality.append(np.linalg.norm(min_eigenvector))

    return directionality


# 其余部分代码保持不变


def get_neighbors(point, point_cloud, radius=0.1):
    # 根据球体半径获取点云中的邻居点

    neighbors = []

    for neighbor in point_cloud:
        distance = np.sqrt(np.sum((point - neighbor) ** 2))  # 计算点之间的欧几里得距离
        if distance <= radius:
            neighbors.append(neighbor)

    return neighbors


# 提供三维点云的路径
file_path = r"G:\text\demo\cloud_point\results\52.txt"
print("读取点云数据中...")

# 从txt文件读取点云数据
with open(file_path, 'r') as f:
    lines = f.readlines()

    # 判断是否有标题行
    if lines[0].startswith('x') or lines[0].startswith('X'):
        point_cloud = np.loadtxt(file_path, skiprows=1)
    else:
        point_cloud = np.loadtxt(file_path)

print("读取点云数据完成。")

directionality = compute_directionality(point_cloud)

# 自适应采样
sampled_indices = []
for i in range(len(point_cloud)):
    if directionality[i] < 0.5:
        sampled_indices.append(i)

sampled_point_cloud = point_cloud[sampled_indices]
print("自适应采样完成。")

# 可视化结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', s=1)
ax.scatter(sampled_point_cloud[:, 0], sampled_point_cloud[:, 1], sampled_point_cloud[:, 2], c='r', s=1)

plt.show()
