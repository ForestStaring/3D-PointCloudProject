import open3d as o3d
import numpy as np
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

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
        return pcd
    else:
        raise ValueError("Unsupported point cloud format")

# 定义点云文件路径
pointcloud_path = r"G:\text\demo\3D-PointCloudProject\My File\output file\hand segmentation leaf .txt"

# 读取点云文件
pcd = read_pointcloud(pointcloud_path)

# 基于密度的聚类算法DBSCAN
clustering = DBSCAN(eps=0.8, min_samples=20).fit(np.asarray(pcd.points))

# 将聚类结果保存到PointCloud对象中
labels = clustering.labels_
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 可视化显示结果
o3d.visualization.draw_geometries([pcd])

# 创建目录用于保存聚类结果
output_dir = "clustered_pointclouds"
os.makedirs(output_dir, exist_ok=True)

# 保存每个聚类的点云
for label in set(labels):
    if label < 0:
        continue

    # 获取该聚类的点云数据
    cluster_points = np.asarray(pcd.points)[labels == label]

    # 构建保存文件路径
    output_path = os.path.join(output_dir, f"cluster_{label}.txt")

    # 将点云数据保存为txt文件
    np.savetxt(output_path, cluster_points, fmt="%.6f", delimiter=" ")

    print(f"Cluster {label} saved to {output_path}")

