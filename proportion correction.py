import open3d as o3d
import numpy as np
import proportion count  # 替换为比例矫正.py的文件名（不包括文件扩展名）
import os

# 定义点云文件路径
pointcloud_path = r"G:\text\demo\3D-PointCloudProject\My File\input file\corn.txt"

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
        if data.shape[1] >= 6:  # 至少包含6列数据，才认为有RGB颜色信息
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255)  # 第4列到第6列为RGB颜色信息，除以255以将值归一化
        return pcd
    else:
        raise ValueError("Unsupported point cloud format")


# 读取点云文件
pcd = read_pointcloud(pointcloud_path)

# 取得之前求得的实际比例
scale_factor = 比例矫正.scale_factor

# 将点云转换为numpy数组
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors) * 255  # 颜色值需要乘以255以将值还原为0-255的整数

# 对点云中的每个点进行坐标缩放
scaled_points = points * scale_factor

# 询问用户要保存的目录
save_dir = r"G:\text\demo\3D-PointCloudProject\My File\output file"

# 生成保存文件的完整路径
file_name = os.path.basename(pointcloud_path)  # 使用输入点云文件的文件名作为保存文件的文件名
file_name = "real_" + file_name  # 添加前缀 "real"
save_path = os.path.join(save_dir, file_name)

# 将点云数据和颜色信息保存为TXT格式的文件
data = np.concatenate((scaled_points, colors), axis=1)
np.savetxt(save_path, data, delimiter=" ", header="X Y Z R G B", comments="")

print(f"处理后的点云数据已保存到 {save_path}")

