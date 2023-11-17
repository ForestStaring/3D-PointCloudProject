import open3d as o3d
import numpy as np
import cv2

# 创建点云对象
pcd = o3d.geometry.PointCloud()

# 读取点云文件
pointcloud_path = r"G:\text\demo\3D-PointCloudProject\clustered_pointclouds\cluster_0.txt"
data = np.loadtxt(pointcloud_path, delimiter=" ", skiprows=1)
pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # 前三列为坐标
if data.shape[1] >= 6:  # 至少包含6列数据，才认为有RGB颜色信息
    pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)  # 第4列到第6列为RGB颜色信息，除以255以将值归一化

# 对点云进行椭圆拟合
points = np.asarray(pcd.points)
ellipse = cv2.fitEllipse(points[:, :2].astype(np.float32))
major_axis = max(ellipse[1])
point_cloud_length = major_axis
print("length:", point_cloud_length)

# 创建椭球体
center = [0, 0, 0]
radii = [1, 2, 3]  # 沿着x、y、z轴的半径

# 创建球体
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
sphere.compute_vertex_normals()

# 缩放球体以创建椭球体
transformation = np.eye(4)
transformation[0, 0] = radii[0]
transformation[1, 1] = radii[1]
transformation[2, 2] = radii[2]
transformed_sphere = sphere.transform(transformation)

# 平移椭球体
transformed_sphere.translate(center)

# 可视化点云和椭球体
o3d.visualization.draw_geometries([pcd, transformed_sphere])
