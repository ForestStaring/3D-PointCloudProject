import open3d as o3d
import numpy as np

# 定义点云文件路径
pointcloud_path = r"G:\text\demo\cloud_point\results\盆子.txt"

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
        data = np.loadtxt(file_path, delimiter=" ")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # 前三列为坐标
        if data.shape[1] == 6:
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])  # 后三列为RGB颜色信息，保留原始值
        return pcd
    else:
        raise ValueError("Unsupported point cloud format")

# 读取点云文件
pcd = read_pointcloud(pointcloud_path)


# 将点云转换为numpy数组
points = np.asarray(pcd.points)  # 单位为点云文件中的单位
colors = np.asarray(pcd.colors)

# 加载盆子真实高度（单位为厘米）
real_h = 19.0  # 假设真实高度为19厘米

# 计算盆子重建算法的高度
bbox = pcd.get_axis_aligned_bounding_box()
reconstructed_h = bbox.max_bound[2] - bbox.min_bound[2]  # 单位为点云文件中的单位

# 计算比例因子
scale_factor = real_h / reconstructed_h  # 将真实高度转换为与点云文件中的单位相同

print("实际的比例为：", scale_factor)


# 将比例因子保存到scale_factor变量
scale_factor_variable_name = 'scale_factor'
globals()[scale_factor_variable_name] = scale_factor