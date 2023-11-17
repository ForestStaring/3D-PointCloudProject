import open3d as o3d
import numpy as np
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
pointcloud_path = r"G:\text\demo\3D-PointCloudProject\My File\output file\Statistical filtering.txt"

# 读取点云文件
pcd = read_pointcloud(pointcloud_path)

# 超体素下采样
voxel_size = 0.05
downsampled_pcd = pcd.voxel_down_sample(voxel_size)

# 条件欧式聚类
clustering = o3d.geometry.PointCloud.cluster_dbscan(
    downsampled_pcd,
    eps=0.5,
    min_points=20
)

# 将聚类结果保存到PointCloud对象中
labels = np.array(clustering)  # 将聚类结果转换为NumPy数组
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
downsampled_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# 可视化显示结果
o3d.visualization.draw_geometries([downsampled_pcd])
