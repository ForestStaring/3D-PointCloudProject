import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取骨架点集的点云数据
file_path = 'G:/text/demo/cloud_point/results/平滑后.txt'
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

# 可视化原始点云和骨架
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 原始点云可视化
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', label='Original Point Cloud')

# 骨架点可视化
skeleton_points = np.array(skeleton_points)
ax.scatter(skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2], c='r', label='Skeleton Points')

# 边集合可视化
for edge in collapsed_edges:
    start = skeleton_points[edge[0]]
    end = skeleton_points[edge[1]]
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], c='g')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()