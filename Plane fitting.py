import open3d as o3d
import numpy as np

# 定义点云文件路径
pointcloud_path = r"G:\text\demo\3D-PointCloudProject\My File\input file\demo.ply"
output_folder = r"G:\text\demo\3D-PointCloudProject\My File\output file"

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
    else:
        raise ValueError("Unsupported point cloud format")

# 读取点云文件
pcd = read_pointcloud(pointcloud_path)

# 将点云转换为numpy数组
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print("->正在RANSAC平面分割...")
distance_threshold = 0.2    # 内点到平面模型的最大距离
ransac_n = 3                # 用于拟合平面的采样点数
num_iterations = 1000       # 最大迭代次数

# 返回模型系数plane_model和内点索引inliers，并赋值
plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)

# 输出平面方程
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 平面内点点云
inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0, 0, 1.0])
print(inlier_cloud)

# 平面外点点云
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([1.0, 0, 0])
print(outlier_cloud)

# 可视化平面分割结果
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

print("->正在删除平面点云区域...")
# 在原始点云中删除平面内的点
pcd_without_plane = pcd.select_by_index(inliers, invert=True)

# 导出剩余点云到文本文件
output_file = output_folder + "/Plane fitting.txt"
np.savetxt(output_file, np.asarray(pcd_without_plane.points), delimiter=" ")

print("->已保存剩余点云到文件:", output_file)

