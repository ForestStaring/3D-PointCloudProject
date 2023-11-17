import open3d as o3d
import numpy as np

# 读取点云文件
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
pointcloud_path = r"G:\text\demo\cloud_point\corrected2.txt"

# 定义超绿针对的颜色通道（RGB三个通道的值均从0-1取值）
color_channel = 1  # 0表示红色，1表示绿色，2表示蓝色

# 定义超绿阈值
thresh_val = 0.6  # 调整阈值以控制判定绿色的严格程度

# 读取点云文件
pcd = read_pointcloud(pointcloud_path)

# 将点云转换为numpy数组
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 提取所选颜色通道的值
color_channel_vals = colors[:, color_channel]

# 进行超绿算法分割
segment_mask = color_channel_vals > thresh_val

# 根据超绿算法分割结果生成新的点云
pcd_seg = o3d.geometry.PointCloud()
pcd_seg.points = o3d.utility.Vector3dVector(points[segment_mask])
pcd_seg.colors = o3d.utility.Vector3dVector(colors[segment_mask])

# 使用欧式聚类方法进行进一步滤波
cl, ind = pcd_seg.remove_radius_outlier(nb_points=20, radius=0.5)  # 调整参数以适应您的场景

# 根据欧式聚类结果生成最终的点云
pcd_filt = pcd_seg.voxel_down_sample(voxel_size=0.2)  # 调整参数以适应您的场景

# 计算 point cloud 的 bounding box
bbox = pcd_filt.get_axis_aligned_bounding_box()

# 获取 bounding box 最小点的坐标
min_pt = bbox.get_min_bound()

# 找到点云中到最小点最远的点
max_dist = 0
for i in range(points.shape[0]):
    dist = np.sqrt(np.sum((points[i]-min_pt)**2))
    if dist > max_dist:
        max_dist = dist
        farthest_point = i
farthest_pt = points[farthest_point]

# 计算法向量
normal = np.cross(farthest_pt-min_pt, np.array([0,0,1]))
normal = normal / np.linalg.norm(normal)

# 计算坐标系原点
origin = min_pt + normal * np.dot(farthest_pt-min_pt, normal)

# 创建坐标系
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=origin)

# 将点云和坐标系合并为一个可视化对象
vis_obj = [pcd_filt, coord_frame]

# 可视化显示结果
o3d.visualization.draw_geometries(vis_obj)

# 定义输出文件路径
output_folder = r"G:\text\demo\cloud_point\results"
output_file_path = output_folder + "\\" + "filtered_pcd.txt"

# 将点云保存为txt格式
points = np.asarray(pcd_filt.points)
colors = np.asarray(pcd_filt.colors)
if colors.size == 0:
    np.savetxt(output_file_path, points, fmt='%f')
else:
    data = np.concatenate((points, colors * 255), axis=1)
    np.savetxt(output_file_path, data, fmt='%f %f %f %d %d %d')

print("Point cloud saved to", output_file_path)
