import open3d as o3d
import numpy as np

def read_pointcloud(file_path):
    extension = file_path.split(".")[-1].lower()

    if extension == "ply":
        return o3d.io.read_point_cloud(file_path)
    elif extension == "pcd":
        return o3d.io.read_point_cloud(file_path)
    elif extension == "obj":
        mesh = o3d.io.read_triangle_mesh(file_path)
        return mesh.sample_points_uniformly(number_of_points=1000000)
    elif extension == "txt":
        data = np.loadtxt(file_path, delimiter=" ", skiprows=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        if data.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)
        if data.shape[1] >= 7:
            pcd.labels = data[:, 6]
        return pcd
    else:
        raise ValueError("Unsupported point cloud format")

# 定义点云文件路径
pointcloud_path = r"G:\text\demo\cloud_point\results\统计滤波后.txt"

# 读取点云文件
pcd = read_pointcloud(pointcloud_path)

# 构造邻域估计
kdtree = o3d.geometry.KDTreeFlann(pcd)
k_neighbors = 80  # 设置K近邻数量
radius = 0.1  # 设置邻域半径

# 获取每个点的邻域点索引
neighbor_indices = []
for i in range(len(pcd.points)):
    [k, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], k_neighbors)
    neighbor_indices.append(idx)

# 构造拉普拉斯加权矩阵
laplacian_matrix = np.zeros((len(pcd.points), len(pcd.points)))
for i in range(len(pcd.points)):
    for j in neighbor_indices[i]:
        laplacian_matrix[i, j] = 1

# 收缩点云
skeleton_points = pcd.points[::10]  # 每隔10个点取一个作为骨架点

# 最远距离采样策略获取骨架顶点
skeleton_indices = [0]  # 初始骨架顶点的索引
while len(skeleton_indices) < len(skeleton_points):
    max_distance = 0
    max_index = 0
    for i in range(len(skeleton_points)):
        if i not in skeleton_indices:
            distances = np.linalg.norm(pcd.points - skeleton_points[i], axis=1)
            min_distance = np.min(distances[skeleton_indices])
            if min_distance > max_distance:
                max_distance = min_distance
                max_index = i
    skeleton_indices.append(max_index)

# 构造边集合
edges = []
for i in range(len(skeleton_indices)):
    for j in range(i + 1, len(skeleton_indices)):
        edges.append([skeleton_indices[i], skeleton_indices[j]])

# 边塌缩操作从边集合中去除冗余边
filtered_edges = []
for edge in edges:
    source = edge[0]
    target = edge[1]
    if laplacian_matrix[source, target] == 1:
        filtered_edges.append(edge)

# 可视化结果
lines = o3d.geometry.LineSet()
lines.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
lines.lines = o3d.utility.Vector2iVector(filtered_edges)
o3d.visualization.draw_geometries([lines])
