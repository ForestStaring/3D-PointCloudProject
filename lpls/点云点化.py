import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fill_point_cloud(point_cloud, threshold_distance, cluster_threshold):
    # 通过DBSCAN聚类算法将点云分成若干个类
    dbscan = DBSCAN(eps=threshold_distance, min_samples=cluster_threshold)
    labels = dbscan.fit_predict(point_cloud)

    # 找到主点云所属的类别
    unique_labels, counts = np.unique(labels, return_counts=True)
    main_cluster_label = unique_labels[np.argmax(counts)]

    # 将主点云的索引保存下来
    main_indices = np.where(labels == main_cluster_label)[0]

    # 初始化断裂点集合
    fractured_indices = np.where(labels != main_cluster_label)[0]
    fractured_points = point_cloud[fractured_indices]

    # 对断裂点进行迭代中间点补全
    while len(fractured_points) > 0:
        new_points = []
        for point in fractured_points:
            # 找到距离主点云最近的点
            nearest_point_index = main_indices[np.argmin(cdist([point], point_cloud[main_indices]))]
            nearest_point = point_cloud[nearest_point_index]

            # 计算中间点
            middle_point = (point + nearest_point) / 2
            new_points.append(middle_point)

        # 将新的中间点添加到主点云中
        point_cloud = np.concatenate((point_cloud, new_points))
        main_indices = np.concatenate((main_indices, np.arange(len(point_cloud) - len(new_points), len(point_cloud))))

        # 重新聚类，更新断裂点集合
        labels = dbscan.fit_predict(point_cloud)
        fractured_indices = np.where(labels != main_cluster_label)[0]
        fractured_points = point_cloud[fractured_indices]

    return point_cloud

# 读取点云文件
def read_point_cloud(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    points = []
    for line in lines[1:]:
        coords = line.strip().split(' ')
        point = [float(coords[i]) for i in range(3)]
        points.append(point)
    return np.array(points)

# 保存点云文件
def save_point_cloud(file_path, point_cloud):
    with open(file_path, 'w') as f:
        f.write("xyzrgb\n")
        for point in point_cloud:
            line = " ".join(str(coord) for coord in point)
            f.write(line + "\n")

# 点云文件路径
file_path = r"G:\text\demo\cloud_point\results\统计滤波后.txt"

# 读取点云数据
point_cloud = read_point_cloud(file_path)

# 设置相邻点的距离阈值和聚类算法的阈值
threshold_distance = 0.01
cluster_threshold = 5

# 补全点云
filled_point_cloud = fill_point_cloud(point_cloud, threshold_distance, cluster_threshold)

# 保存补全后的点云到新文件
output_file_path = "1.txt"
save_point_cloud(output_file_path, filled_point_cloud)

# 可视化点云
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filled_point_cloud[:,0], filled_point_cloud[:,1], filled_point_cloud[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
