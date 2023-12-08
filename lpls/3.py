import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev


# 读取骨架点集的点云数据
file_path = 'G:/text/demo/cloud_point/results/42.txt'
point_cloud = np.loadtxt(file_path)


# 最远距离采样策略：根据点云中的点与其他点之间的最大距离来进行采样
def farthest_point_sampling(points, num_samples):
    sampled_points = []
    sampled_indices = []
    sampled_points.append(points[0])
    sampled_indices.append(0)

    for i in range(1, num_samples):
        distances = np.linalg.norm(points - np.array(sampled_points)[:, np.newaxis], axis=-1)
        min_distances = np.min(distances, axis=0)
        max_distance_index = np.argmax(min_distances)
        sampled_points.append(points[max_distance_index])
        sampled_indices.append(max_distance_index)

    return sampled_points, sampled_indices


# 从骨架点集中进行最远距离采样
num_samples = 100  # 需要采样的骨架点数
skeleton_points, sampled_indices = farthest_point_sampling(point_cloud, num_samples)


# 连接相邻的骨架顶点，生成边集合
def connect_adjacent_points(points):
    edges = []
    num_points = len(points)

    for i in range(num_points - 1):
        for j in range(i + 1, num_points):
            edges.append((i, j))

    return edges


# 根据骨架点集生成边集合
edges = connect_adjacent_points(skeleton_points)


# 边塌缩操作：去除冗余边
def edge_collapse(edges):
    # 这里需要根据你具体的边塌缩操作实现
    # ...

    return edges


# 执行边塌缩操作
collapsed_edges = edge_collapse(edges)

# 输出最终的骨架点和边集合
print("骨架点集:")
for point in skeleton_points:
    print(point)

print("边集合:")
for edge in collapsed_edges:
    print(edge)



# 将skeleton_points转换为NumPy数组
skeleton_points = np.array(skeleton_points)

# 平滑处理
tck, u = splprep([skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2]], s=20)
smooth_points = np.transpose(splev(u, tck))

# 创建骨架可视化窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制平滑后的骨架点
ax.scatter(smooth_points[:, 0], smooth_points[:, 1], smooth_points[:, 2], c='r', marker='o')

# 连接相邻的骨架点
for i in range(len(collapsed_edges)):
    start = collapsed_edges[i][0]
    end = collapsed_edges[i][1]
    if np.linalg.norm(skeleton_points[start] - skeleton_points[end]) < 0.5:
        ax.plot([smooth_points[start][0], smooth_points[end][0]],
                [smooth_points[start][1], smooth_points[end][1]],
                [smooth_points[start][2], smooth_points[end][2]], c='g')

plt.show()