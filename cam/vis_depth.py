import numpy as np
import cv2

# 加载 .npy 文件
depth_data = np.load('0_depth.npy')

# 指定四个关键点
rgb_keypoints = np.array([
    [101, 83],
    [422, 83],
    [101, 410],
    [422, 410]
])

# 获取矩形区域的最小和最大坐标
x_min = np.min(rgb_keypoints[:, 0])
x_max = np.max(rgb_keypoints[:, 0])
y_min = np.min(rgb_keypoints[:, 1])
y_max = np.max(rgb_keypoints[:, 1])

# 裁剪深度图到指定区域
depth_region = depth_data[y_min:y_max, x_min:x_max]

# 归一化深度图到 0-255 范围
depth_normalized = cv2.normalize(depth_region, None, 0, 255, cv2.NORM_MINMAX)

# 将深度图转换为 uint8 格式
depth_uint8 = depth_normalized.astype(np.uint8)

# 使用 opencv 显示灰度深度图
cv2.imshow('Cropped Depth Map (Grayscale)', depth_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()



