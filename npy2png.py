import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d

root_path = "semantic_3d_pointcloud/"

point = np.load(os.path.join(root_path, "point.npy"))
point = point*10000/255.
color = np.load(os.path.join(root_path, "color01.npy"))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point)
pcd.colors = o3d.utility.Vector3dVector(color)

points = np.asarray(pcd.points)
pcd = pcd.select_by_index(np.where(points[:, 1] < 0)[0])
points = np.asarray(pcd.points)
pcd = pcd.select_by_index(np.where(points[:, 1] > -1)[0])
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

px = 1/plt.rcParams['figure.dpi']
plt.figure(figsize=(960*px, 720*px))
plt.axis('off')
plt.scatter(-points[:, 2], -points[:, 0], s=1, c=colors, alpha=0.5)
plt.savefig("map.png")
# o3d.visualization.draw_geometries([pcd])