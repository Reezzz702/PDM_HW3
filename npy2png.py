import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

root_path = "semantic_3d_pointcloud/"

point = np.load(os.path.join(root_path, "point.npy"))
point = point*10000/255.
color = np.load(os.path.join(root_path, "color0255.npy"))
color = color/255.
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point)
pcd.colors = o3d.utility.Vector3dVector(color)

points = np.asarray(pcd.points)
pcd = pcd.select_by_index(np.where(points[:, 1] < 0)[0])
points = np.asarray(pcd.points)
pcd = pcd.select_by_index(np.where(points[:, 1] > -1)[0])
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# px = 1/plt.rcParams['figure.dpi']
# plt.figure(figsize=(960*px, 720*px))
# plt.axis('off')
# plt.scatter(-points[:, 2], -points[:, 0], s=1, c=colors, alpha=0.5)
fig = plt.figure()
map = fig.add_subplot()
map.scatter(points[:, 2], points[:, 0], s=1, c=colors, alpha=0.5)
# coords = []

# def onclick(event):
#     global ix, iy
#     ix, iy = event.xdata, event.ydata
#     print ('x = %d, y = %d'%(
#         ix, iy))

#     global coords
#     coords.append((ix, iy))

#     if len(coords) == 2:
#         fig.canvas.mpl_disconnect(cid)

#     return coords
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# print(coords)
plt.show()
plt.savefig("map.png")
img = cv2.imread("map.png")
cv2.imshow("figure", img)
cv2.waitKey(0)
# o3d.visualization.draw_geometries([pcd])