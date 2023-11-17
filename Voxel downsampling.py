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
pointcloud_path = r"G:\text\demo\3D-PointCloudProject\My File\output file\Statistical filtering.txt"

print("->正在加载点云... ")
# 读取点云文件
pcd = read_pointcloud(pointcloud_path)

# 打印原始点云总数
print("原始点云总数：", len(pcd.points))

print("->正在体素下采样...")
voxel_size = 0.1
downpcd = pcd.voxel_down_sample(voxel_size)
print(downpcd)

# 打印剩余点云总数
print("剩余点云总数：", len(downpcd.points))

print("->正在可视化下采样点云")
o3d.visualization.draw_geometries([downpcd], width=800, height=600)


