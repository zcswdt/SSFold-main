import numpy as np

intrinsics = np.array([[604.99621582, 0.0, 329.07382202],
                       [0.0, 605.07873535, 243.38549805],
                       [0.0, 0.0, 1.0]])

#读取变换矩阵cam_pose
tx_world_camera = np.loadtxt(
    '/home/zcs/work/github/folding-by-hand/calibration/cam2table_pose_2.txt')

tx_robot = np.loadtxt(
    '/home/zcs/work/github/folding-by-hand/calibration/cam_pose.txt')

#像素坐标转-相机坐标
# 相机内参
f_x = intrinsics[0, 0]  # 焦距 f_x
c_x = intrinsics[0, 2]  # 主点坐标 c_x
f_y = intrinsics[1, 1]  # 焦距 f_y
c_y = intrinsics[1, 2]  # 主点坐标 c_y

#[695.4378 292.1603]
# 给定的像素坐标(u, v)和深度Z
u = 262 # 填入具体值
v =238 # 填入具体值
# u = 433#262 # 填入具体值
# v =125#238
Z = 0.878 # 从深度相机获得的深度信息
# 将像素坐标转换为归一化图像坐标
x = (u - c_x) / f_x
y = (v - c_y) / f_y

# 使用深度信息获得相机坐标系下的三维坐标
X = x * Z
Y = y * Z
# X = x
# Y = y
#Z值不变

print(f"相机坐标系下的点坐标: ({X}, {Y}, {Z})")
# # (-0.09722993363158722, -0.007805730913875184, 0.877)


# 之前计算得到的相机坐标系下的点
camera_coordinates = np.array([X, Y, Z, 1])  # 齐次坐标

# 使用矩阵乘法将相机坐标转换为世界坐标
world_coordinates = np.dot(tx_world_camera, camera_coordinates)

# 从齐次坐标转换回3D坐标
world_coordinates = world_coordinates[:3]






#世界坐标系-相机坐标系
import numpy as np
def transform_point(point, transformation_matrix):
    """应用变换矩阵来转换点"""
    point_homogeneous = np.append(point, 1)  # 将点转换为齐次坐标
    transformed_point_homogeneous = np.dot(transformation_matrix, point_homogeneous)  # 应用变换矩阵
    return transformed_point_homogeneous[:3]  # 返回转换后的三维坐标



# 桌面点C的坐标（假设Z=0，即点在桌面上）
C_desk = np.array([0, 0, 0])
#Step 1: 桌面点转换到相机坐标系
C_cam = transform_point(C_desk, np.linalg.inv(tx_world_camera))  # 使用M_cam2table的逆进行转换
print(f"相机坐标系下的点坐标:",C_cam)
#0.01,0.063,0.0865





# 逆矩阵
A_inv = np.linalg.inv(tx_robot)
x=0.35
y=-0.15
z=0.2
# 机械臂坐标系中的点
point_arm = np.array([x, y, z, 1])  # 确保使用正确的z值

# 转换到相机坐标系
point_camera = np.dot(A_inv, point_arm)
print("Point in camera coordinates:", point_camera[:3])

# 使用B矩阵将相机坐标转换到世界坐标系
point_world = np.dot(tx_world_camera, point_camera)
print("Point in world coordinates:", point_world[:3])