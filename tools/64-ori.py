import cv2
import numpy as np

# 已知的关键点
crop_keypoints = np.array([
    [0, 0],
    [300, 0],
    [0, 300],
    [300, 300],
])

rgb_keypoints = np.array([
    [261, 80],
    [566, 77],
    [260, 380],
    [566, 383]
])

# 读取变换后的图像 (64x64)
color_resized = cv2.imread('color_resized.jpg')

# 先将变换后的图像从 64x64 恢复到 300x300
color_resized_to_300 = cv2.resize(color_resized, (300, 300))
# 保存恢复的原始图像
cv2.imwrite('color_resized_to_300_recovered.jpg', color_resized_to_300)
# 计算单应性矩阵
rgb_h, _ = cv2.findHomography(rgb_keypoints, crop_keypoints)

# 计算逆单应性矩阵
inv_rgb_h = np.linalg.inv(rgb_h)

# 使用逆单应性矩阵对图像进行逆透视变换
img_recovered = cv2.warpPerspective(color_resized_to_300, inv_rgb_h, (640, 480))

# 保存恢复的原始图像
cv2.imwrite('img_recovered.jpg', img_recovered)

# 显示结果
cv2.imshow('Recovered Image', img_recovered)
cv2.imshow('Color Resized to 300', color_resized_to_300)
cv2.waitKey(0)
cv2.destroyAllWindows()


