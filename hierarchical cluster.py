import open3d as o3d
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# 读取点云文件
def read_pointcloud(file_path):
    extension = file_path.split(".")[-1].lower()

    if extension == "ply":
        return o3d.io.read_point_cloud(file_path)
    elif extension == "pcd":
        return o3d.io.read_point_cloud(file_path)
    elif extension == "obj":
        mesh = o3d.io.read_triangle_mesh(file_path)
        return mesh.sample_points_uniformly(number_of_points=1000000)  # 将OBJ文件转换为点云采样
    elif extension == "txt":
        data = np.loadtxt(file_path, delimiter=" ", skiprows=1)  # 跳过第一行标题行
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # 前三列为坐标
        if data.shape[1] >= 6:  # 至少包含6列数据，才认为有RGB颜色信息
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)  # 第4列到第6列为RGB颜色信息，除以255以将值归一化
        if data.shape[1] >= 7:  # 至少包含7列数据，才认为有常量标签信息
            pcd.labels = data[:, 6]  # 第7列为常量标签信息
        return pcd
    else:
        raise ValueError("Unsupported point cloud format")

# 定义点云文件路径
pointcloud_path = r"G:\text\demo\3D-PointCloudProject\My File\output file\Statistical filtering.txt"

# 读取点云文件
pcd = read_pointcloud(pointcloud_path)

# 获取点云数据并进行排序
points = np.asarray(pcd.points)
centroid = np.mean(points, axis=0)
points -= centroid
cov = np.dot(points.T, points) / points.shape[0]
eigen_values, eigen_vectors = np.linalg.eigh(cov)
growth_direction = eigen_vectors[:, 2]  # 取最小的特征值对应的特征向量作为生长方向

# 分层聚类
clustering = AgglomerativeClustering(n_clusters=5).fit(points)

# 可视化聚类结果
colors = [1, 0, 0], [0, 1, 0], [0, 0, 1],[0, 1, 1], [1, 0, 1] # 定义三种颜色
color_arr = np.zeros_like(points)
for i in range(0,5):
    color_arr[clustering.labels_ == i] = colors[i]
pcd.colors = o3d.utility.Vector3dVector(color_arr)

# 将点云可视化以验证聚类效果
o3d.visualization.draw_geometries([pcd])

