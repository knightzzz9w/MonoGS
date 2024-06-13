from scipy.spatial.transform import Rotation as R
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import math
#test1
# quaternion = [0.0, 2.0, 0.0, 2.0]
# rotation_matrix = R.from_quat(quaternion).as_matrix()
# print(rotation_matrix)


#test2

# def getProjectionMatrix(znear, zfar, cx, cy, fx, fy, W, H):
#     P = np.zeros((4, 4))

#     P[0, 0] = 2.0 * fx / W
#     P[1, 1] = 2.0 * fy / H
#     P[0, 2] = 2.0 * cx/W -1
#     P[1, 2] = 2.0 * cy/H -1
#     P[3, 2] = 1.0
#     P[2, 2] =  zfar / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)

#     return P



# def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
#     left = ((2 * cx - W) / W - 1.0) * W / 2.0
#     right = ((2 * cx - W) / W + 1.0) * W / 2.0
#     top = ((2 * cy - H) / H + 1.0) * H / 2.0
#     bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
#     left = znear / fx * left
#     right = znear / fx * right
#     top = znear / fy * top
#     bottom = znear / fy * bottom
#     P = np.zeros((4, 4))

#     z_sign = 1.0

#     P[0, 0] = 2.0 * znear / (right - left)
#     P[1, 1] = 2.0 * znear / (top - bottom)
#     P[0, 2] = (right + left) / (right - left)
#     P[1, 2] = (top + bottom) / (top - bottom)
#     P[3, 2] = z_sign
#     P[2, 2] = z_sign * zfar / (zfar - znear)
#     P[2, 3] = -(zfar * znear) / (zfar - znear)

#     return P


# znear = 0.01 ; zfar = 100 ; cx = 100 ; cy = 102; fx = 307; fy = 430; W = 2013;  H  = 3100
# K = np.array([[fx , 0 , cx] , [0 , fy , cy] ,[0,0,1] ] , dtype=float)

# x_ext = np.array([0.7,0.6,50,1.]) ; x_origin = np.array([0.7,0.6,50])
# #P1 = getProjectionMatrix(znear, zfar, cx, cy, fx, fy, W, H)


# P1 = getProjectionMatrix(znear, zfar, cx, cy, fx, fy, W, H)
# xP1= P1@x_ext.reshape(-1,1)
# x1 = xP1[:3]/xP1[3]
# x_cor1 = [(x1[0] + 1)*0.5*W , (x1[1] + 1)*0.5*H]
# print("xcor 1 is " , x_cor1)


# P2 = getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H)
# xP2 = P2@x_ext.reshape(-1,1)
# x2 = xP2[:3]/xP2[3]
# x_cor2 = [(x2[0] + 1)*0.5*W , (x2[1] + 1)*0.5*H]
# print("xcor 2 is " , x_cor2)

# x_K = K@x_origin.reshape(-1,1)
# print("xK is " , x_K)
# x_cor2 = [x_K[0]/x_K[2] , x_K[1]/x_K[2]]
# print("xcor2 is " , x_cor2)


#test3

# import numpy as np
# import cv2

# img_path = "/home/wkx123/MonoGS/datasets/euroc/MH_01_easy/mav0/cam0/data/1403636579763555584.png"
# image = cv2.imread(img_path)
# print(image.shape)
# print(image[100:110,100:110,0])
# print(image[100:110,100:110,1])
# print(image[100:110,100:110,2])

#test4

# def getpicdepth( points_2D , init_depth_map , height , width , K):
    
#     # 初始化深度图
#     depth_map = np.full((height, width), np.nan)

#     # 提取深度值
#     depths = init_depth_map
    
#     # 将已知深度值填入深度图
#     for idx, (x, y) in enumerate(points_2D):
#         depth_map[y,x] = depths[idx]

#     # 构建KNN模型
#     t_knn1 = time.time()
#     knn = NearestNeighbors(n_neighbors=1)
#     knn.fit(points_2D)
#     t_knn2 = time.time()

#     print("knn construct time is " , t_knn2 - t_knn1)
#     # 创建一个新的深度图以存储结果
#     new_depth_map = np.copy(depth_map)

#     t_knn1 = time.time()
#     # 遍历所有像素点
#     for i in range(height):
#         for j in range(width):
#             if np.isnan(new_depth_map[i, j]):  # 如果点没有初始深度信息
#                 # 获取半径为3的邻域
#                 top = max(0, i-20) ; bot = min(height, i+21) ; left = max(0, j-20) ; right = min(width, j+21)
#                 neighborhood = new_depth_map[top:bot, left:right]
#                 if np.all(np.isnan(neighborhood)):  #如果这个点没有被填充
#                     # 使用KNN计算该点的深度值
#                     distances, indices = knn.kneighbors([[j, i]])
#                     neighbor_depths = depths[indices[0]]
#                     weights = 1 / (distances[0] + 1e-8)  # 防止除零
#                     weighted_sum = np.sum(weights * neighbor_depths)
#                     sum_weights = np.sum(weights)
#                     new_depth_map[top:bot, left:right] = 0  # 标记为这个区域已填充 但是这些点是没有深度的 只有一个地方有深度
#                     new_depth_map[i, j] = weighted_sum / sum_weights

#     # 使用 cv2.inpaint 修复深度图中的空缺点
#     t_knn2 = time.time()
    
    
#     print("knn finding  time is " , t_knn2 - t_knn1)
#     new_depth_map[np.isnan(new_depth_map)] = 0  # 将 NaN 替换为 0
#     #new_depth_map_filled = cv2.inpaint(new_depth_map.astype(np.float32), mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

#     u, v = np.meshgrid(np.arange(width), np.arange(height))
#     u = u.flatten()
#     print(u)
#     v = v.flatten()
#     homogeneous_pixel_coords = np.vstack((u, v, np.ones_like(u)))
#     print(homogeneous_pixel_coords.shape)
    
#     depths = new_depth_map.flatten()
#     valid_mask = depths > 0
#     valid_pixel_coords = homogeneous_pixel_coords[:, valid_mask]
#     valid_depths = depths[valid_mask]
    
    
#     cam_coords = np.linalg.inv(K) @ (valid_pixel_coords * valid_depths)
#     new_points_2D = valid_pixel_coords[:2].T
#     new_points_3D_cam = cam_coords.T

#     return new_points_2D , new_points_3D_cam


# width = 1000 ; height = 600

# array1 = np.arange(0, width , 10) ; array2 = np.arange(0 , height , 10)
# points_2D = np.array([[i,j] for i in array1 for j in array2])
# print(points_2D.shape)
# init_depth_map = np.ones(points_2D.shape[0])

# print("point before is " , points_2D.shape[0])

# K = np.array([[400 , 0 , 50] , [0 ,  400  , 50] , [0 , 0 , 1] ])
# t1 = time.time()
# new2d , new3d = getpicdepth( points_2D , init_depth_map , height , width , K)
# print("point after is " , new2d.shape[0])
# t2 = time.time()
# print(t2 - t1)



# print(math.atan(1))
# print(math.pi/4)



# a = np.array([[0.1 , 0.2, 0.3] , [0.4 , 0.5, 0.6] , [0.7 , 0.8, 0.9]])
# b = np.array([[True , False, True] , [False , False, True] , [True , False, True]])
# c = a[b]
# c[0] = 1
# print(a)
# ind1 = np.array([0 , 2 , 1 , 1 , 2])
# ind2 = np.array([0 , 0 , 1 , 2 ,1])
# print(a[ind1 , ind2])


import numpy as np
import torch
import time

# 设置矩阵大小
matrix_size = 10000

# 使用numpy在CPU上计算
a_np = np.random.rand(matrix_size, matrix_size)
b_np = np.random.rand(matrix_size, matrix_size)

start_time = time.time()
result_np = np.dot(a_np, b_np)
end_time = time.time()

print("NumPy (CPU) computation time: {:.4f} seconds".format(end_time - start_time))

# 使用PyTorch在GPU上计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a_torch = torch.rand(matrix_size, matrix_size, device=device)
b_torch = torch.rand(matrix_size, matrix_size, device=device)

start_time = time.time()
result_torch = torch.mm(a_torch, b_torch)
torch.cuda.synchronize()  # 等待所有CUDA操作完成
end_time = time.time()

print("PyTorch (GPU) computation time: {:.4f} seconds".format(end_time - start_time))

# 使用PyTorch在CPU上计算
a_torch_cpu = torch.rand(matrix_size, matrix_size)
b_torch_cpu = torch.rand(matrix_size, matrix_size)

start_time = time.time()
result_torch_cpu = torch.mm(a_torch_cpu, b_torch_cpu)
end_time = time.time()

print("PyTorch (CPU) computation time: {:.4f} seconds".format(end_time - start_time))
