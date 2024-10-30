import cv2
import numpy as np

# 原始图像的宽度和高度
img_width, img_height = 640, 480

# 读取原始图像
img = cv2.imread('img/ir_origin.png')  # 请将 'path_to_original_image.jpg' 替换为你的原始图像路径

# 已知的关键点
crop_keypoints = np.array([
    [0, 0],
    [300, 0],
    [0, 300],
    [300, 300],
])

# rgb_keypoints = np.array([
#     [117, 72],
#     [444, 72],
#     [117, 401],
#     [443, 402]
# ])


rgb_keypoints = np.array([
    [167, 104],
    [411, 104],
    [168, 346],
    [409, 349]
])
# 计算单应性矩阵
rgb_h, _ = cv2.findHomography(rgb_keypoints, crop_keypoints)

# 使用单应性矩阵对原始图像进行透视变换
color = cv2.warpPerspective(img, rgb_h, (300, 300))
cv2.imwrite('color300.jpg', color)
# 将变换后的图像调整为 64x64
color_resized = cv2.resize(color, (64, 64))

# 保存变换后的图像
cv2.imwrite('color_resized.jpg', color_resized)


