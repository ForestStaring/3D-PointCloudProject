import open3d as o3d
import numpy as np
from scipy.sparse.csgraph import dijkstra

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
pointcloud_path = r"G:\text\demo\cloud_point\results\统计滤波后.txt"

# 读取点云文件
pcd = read_pointcloud(pointcloud_path)

# 将点云转换为numpy数组
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 沿边界裁剪骨架点集
def clip_skeleton_by_boundary(skeleton, bbox):

    print("->正在将骨架点集沿边界裁剪...")
    bbox_min, bbox_max = bbox.get_min_bound(), bbox.get_max_bound()
    bbox_center, bbox_extent = bbox.get_center(), bbox.get_max_bound() - bbox.get_min_bound()
    bbox_extent *= 1.1  # 这里为了让边界稍微扩大一点，避免误差

    cropped_skeleton = o3d.geometry.PointCloud()
    cropped_indices = []
    for i in range(len(skeleton.points)):
        p = skeleton.points[i]
        if (p[0] >= bbox_min[0] and p[1] >= bbox_min[1] and p[2] >= bbox_min[2] and
            p[0] <= bbox_max[0] and p[1] <= bbox_max[1] and p[2] <= bbox_max[2]):
            cropped_indices.append(i)

    cropped_skeleton.points = o3d.utility.Vector3dVector(np.asarray(skeleton.points)[cropped_indices])
    cropped_skeleton.paint_uniform_color([1, 0.706, 0])

    clipped_skeleton = cropped_skeleton.crop(bbox)
    clipped_skeleton.translate(-bbox_center)

    print("->成功将骨架点集沿边界裁剪。")
    return clipped_skeleton



print("->正在估计法线并可视化...")
radius = 0.01   # 搜索半径
max_nn = 30     # 邻域内用于估算法线的最大点数
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))     # 执行法线估计
#o3d.visualization.draw_geometries([pcd], point_show_normal=True)

print("->正在打印前10个点的法向量...")
print(np.asarray(pcd.normals)[:10, :])

# 将点云转换为网格模型
print("->正在将点云转换为网格模型...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0)
mesh.compute_vertex_normals()
mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) + np.asarray(mesh.vertex_normals) * radius)
triangles = np.asarray(mesh.triangles)

# 进一步处理骨架点集
simplified_mesh = mesh.simplify_quadric_decimation(100000)
simplified_mesh = simplified_mesh.filter_smooth_taubin(number_of_iterations=5)
simplified_mesh.compute_vertex_normals()

# 将骨架点集转换回点云数据
skeleton_points = np.asarray(simplified_mesh.vertices)

# 创建新的点云对象
skeleton_pcd = o3d.geometry.PointCloud()
skeleton_pcd.points = o3d.utility.Vector3dVector(skeleton_points)

# 裁剪骨架点集
clipped_skeleton = clip_skeleton_by_boundary(skeleton_pcd, skeleton_pcd.get_axis_aligned_bounding_box())

# 可视化结果
o3d.visualization.draw_geometries([clipped_skeleton])