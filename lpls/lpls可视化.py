import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csgraph
import open3d as o3d

def visualization(np_matrix):
    # 可以根据实际情况对smoothed_point_cloud进行可视化或者其他进一步处理
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np_matrix)
    o3d.visualization.draw_geometries([point_cloud])

def laplace(source_data, n_neighbors, shrinkage_factor):
    point_cloud = source_data
    # 构建邻接关系
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)
    # 构建邻接矩阵
    adjacency_matrix = np.zeros((len(point_cloud), len(point_cloud)))
    for i, point_indices in enumerate(indices):
        adjacency_matrix[i, point_indices] = 1
        adjacency_matrix[point_indices, i] = 1

    # 计算拉普拉斯矩阵
    laplacian_matrix = csgraph.laplacian(adjacency_matrix, normed=False)

    # 对拉普拉斯矩阵进行收缩操作
    shrinked_laplacian = laplacian_matrix + shrinkage_factor * np.identity(len(point_cloud))

    # 从收缩后的拉普拉斯矩阵中恢复平滑化后的点云数据
    smoothed_point_cloud = np.linalg.pinv(shrinked_laplacian).dot(point_cloud)
    print(np.asarray(smoothed_point_cloud).shape)
    return smoothed_point_cloud

def point2numpy(point):
    xyz_load = np.asarray(point.points)
    return xyz_load

point = o3d.io.read_point_cloud(r"G:\text\demo\cloud_point\results\51.txt", format="xyz")
source_data = point2numpy(point)

n_neighbors = 100  # 选择每个点的邻居数
shrinkage_factor = 1  # 设定收缩因子
smoothed_point_cloud = laplace(source_data, n_neighbors, shrinkage_factor)

# 将平滑后的点云保存到指定文件夹
output_file = r"G:\text\demo\cloud_point\results\52.txt"
np.savetxt(output_file, smoothed_point_cloud)


def visualization(original_points, smoothed_points):
    # 创建原始点云和平滑后点云对象
    original_cloud = o3d.geometry.PointCloud()
    original_cloud.points = o3d.utility.Vector3dVector(original_points)
    original_cloud.paint_uniform_color([1, 0, 0])  # 设置原始点云颜色为红色

    smoothed_cloud = o3d.geometry.PointCloud()
    smoothed_cloud.points = o3d.utility.Vector3dVector(smoothed_points)
    smoothed_cloud.paint_uniform_color([0, 0, 1])  # 设置平滑后点云颜色为蓝色

    # 合并两个点云对象
    merged_cloud = original_cloud + smoothed_cloud

    # 显示配准后的点云
    o3d.visualization.draw_geometries([merged_cloud])

# 可视化平滑后的点云和原始点云
visualization(source_data, smoothed_point_cloud)


# 可视化平滑后的点云
#visualization(smoothed_point_cloud)
