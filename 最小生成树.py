import open3d as o3d
import numpy as np

# 读取三维点云数据
shrinked_point_cloud = o3d.io.read_point_cloud("G:/text/demo/cloud_point/results/平滑后.txt", format='xyz')

# 最近邻搜索，构建kdtree
kdtree = o3d.geometry.KDTreeFlann(shrinked_point_cloud)

# 设置最近邻的数量
k = 10

# 对于每个点，找到它的最近邻点
lines = []
for i in range(len(shrinked_point_cloud.points)):
    [k, idx, _] = kdtree.search_knn_vector_3d(shrinked_point_cloud.points[i], k)
    for j in range(1, k):
        lines.append([i, idx[j]])

# 创建线元素
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(np.asarray(shrinked_point_cloud.points)),
    lines=o3d.utility.Vector2iVector(lines),
)

# 可视化最近邻线
o3d.visualization.draw_geometries([shrinked_point_cloud, line_set])

