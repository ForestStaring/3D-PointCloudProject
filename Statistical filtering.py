import open3d as o3d
import numpy as np

# 读取点云文件
def read_pointcloud(file_path, keep_colors=True):
    extension = file_path.split(".")[-1].lower()

    if extension == "ply":
        pcd = o3d.io.read_point_cloud(file_path)
    elif extension == "pcd":
        pcd = o3d.io.read_point_cloud(file_path)
    elif extension == "obj":
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_uniformly(number_of_points=1000000) # 将OBJ文件转换为点云采样
    elif extension == "txt":
        data = np.loadtxt(file_path, delimiter=" ", skiprows=1)  # 跳过第一行标题行
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # 前三列为坐标
        if data.shape[1] >= 6 and keep_colors:  # 至少包含6列数据，才认为有RGB颜色信息
            colors = data[:, 3:6] / 255  # 第4列到第6列为RGB颜色信息，除以255以将值归一化
            pcd.colors = o3d.utility.Vector3dVector(colors)

    else:
        raise ValueError("Unsupported point cloud format")

    return pcd

# 定义点云文件路径
pointcloud_path = r"G:\text\demo\3D-PointCloudProject\My File\output file\super green.txt"

print("->正在加载点云... ")
# 读取点云文件
pcd = read_pointcloud(pointcloud_path, keep_colors=True)

print("->正在进行统计滤波...")
num_neighbors = 100 # K邻域点的个数
std_ratio = 1.0 # 标准差乘数

# 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
sor_pcd, ind = pcd.remove_statistical_outlier(num_neighbors, std_ratio)
print("统计滤波后的点云：", sor_pcd)
# 提取噪声点云
sor_noise_pcd = pcd.select_by_index(ind,invert = True)
print("噪声点云：", sor_noise_pcd)

# 将噪声点云设置为蓝色
sor_noise_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # RGB=0, 0, 255

# 可视化统计滤波后的点云和噪声点云
o3d.visualization.draw_geometries([sor_pcd, sor_noise_pcd])


# 从自定义字段中获取RGB信息
if hasattr(sor_pcd, "colors"):
    rgb = np.asarray(sor_pcd.colors) * 255  # 将RGB值还原到0-255范围
    output_data = np.hstack((sor_pcd.points, np.zeros((len(sor_pcd.points), 3))))  # 创建与坐标维度相同的空的RGB信息
    output_data[:, 3:6] = rgb  # 替换空的RGB信息为真实的RGB值
else:
    output_data = np.asarray(sor_pcd.points)

# 将点云保存为TXT文件
output_file = r"G:\text\demo\3D-PointCloudProject\My File\output file\Statistical filtering.txt"
np.savetxt(output_file, output_data, fmt="%f %f %f %d %d %d", delimiter=" ", header="x y z r g b", comments="")



