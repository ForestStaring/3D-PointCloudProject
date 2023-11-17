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
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)  # 第4列到第6列为RGB颜色信息，除以255以将值归一化
    else:
        raise ValueError("Unsupported point cloud format")

    return pcd

# 定义点云文件路径
pointcloud_path = r"G:\text\demo\3D-PointCloudProject\My File\output file\Statistical filtering.txt"

print("->正在加载点云... ")
# 读取点云文件
pcd = read_pointcloud(pointcloud_path, keep_colors=True)

print("->正在计算点云轴向最小包围盒...")
# 计算点云轴向最小包围盒
aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)

# 计算点云方向最小包围盒
obb = pcd.get_oriented_bounding_box()
obb.color = (0, 1, 0)

# 比较包围盒体积
aabb_volume = aabb.volume()
obb_volume = obb.volume()

if aabb_volume < obb_volume:
    print("-> 点云轴向最小包围盒是最小的")
    print("轴向最小包围盒的大小：")
    print(aabb.max_bound - aabb.min_bound)
else:
    print("-> 点云方向最小包围盒是最小的")
    print("方向最小包围盒的大小：")
    print(obb.extent)

o3d.visualization.draw_geometries([pcd, aabb, obb])

