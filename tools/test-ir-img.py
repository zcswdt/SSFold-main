import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
def get_mask(img, ir):
    # 将红外图像转换为灰度图
    ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    # 创建掩码，其中红外强度大于20的部分为True
    mask = ir > 20
    return mask

def apply_mask_to_depth(depth, mask):
    # 归一化深度图
    normalized_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    # 转换为 uint8 类型
    normalized_depth = normalized_depth.astype(np.uint8)
    # 创建一个全为0的数组，形状与深度图相同
    masked_depth = np.zeros_like(normalized_depth)
    # 掩码区域内设置为255（而非1）以便在图像中明显显示
    masked_depth[mask] = 255
    return masked_depth


# 关键点数组，用于单应性变换
crop_keypoints = np.array([
    [0, 0],
    [300, 0],
    [0, 300],
    [300, 300]
])
rgb_keypoints = np.array([
    [101, 83],
    [422, 83],
    [101, 410],
    [422, 410]
])

ir_keypoints = np.array([
    [156, 114],
    [393, 114],
    [156, 353],
    [393, 353]
])

#
# # 关键点数组，用于单应性变换
# crop_keypoints = np.array([
#     [0, 0],
#     [300, 0],
#     [0, 300],
#     [300, 300]
# ])
# rgb_keypoints = np.array([
#     [87, 69],
#     [431, 69],
#     [87, 416],
#     [431, 416]
# ])
#
# ir_keypoints = np.array([
#     [146, 103],
#     [402, 103],
#     [146, 357],
#     [402, 357]
# ])

#
#
# rgb_keypoints = np.array([
#     [261, 80],
#     [566, 77],
#     [260, 380],
#     [566, 383]
# ])
#
# ir_keypoints = np.array([
#     [298, 102],
#     [523, 100],
#     [298, 325],
#     [522, 325]
# ])

# 计算单应性矩阵
h_ir, _ = cv2.findHomography(ir_keypoints, crop_keypoints)
h_rgb, _ = cv2.findHomography(rgb_keypoints, crop_keypoints)

# 读取图像和深度图
n_file = '/home/zcs/work/github/folding-by-hand/cam/saved_images/'
output_dir = '/home/zcs/work/github/folding-by-hand/cam/saved_images/out1'



o_rgb = cv2.imread(n_file + "color.png")
o_ir = cv2.imread(n_file + "ir.png")
o_dep = np.load(n_file + "depth.npy")

# 应用单应性变换
o_i = cv2.warpPerspective(o_ir, h_ir, (300, 300))
o_de = cv2.warpPerspective(o_dep, h_ir, (300, 300))
o_rg = cv2.warpPerspective(o_rgb, h_rgb, (300, 300))

# 使用红外图像获取掩码
o_mask = get_mask(o_rg, o_i)
o_rg[o_mask == False] = 0


# 应用掩码到深度图，掩码区域内为1，外部为0
o_de_masked = apply_mask_to_depth(o_de, o_mask)



# 保存处理后的图像
cv2.imwrite(os.path.join(output_dir, 'masked_rgb.png'), o_rg)
cv2.imwrite(os.path.join(output_dir, 'masked_depth.png'), o_de_masked)
cv2.imwrite(os.path.join(output_dir, 'ir_image.png'), o_i)

print("Images have been saved to:", output_dir)
# 使用matplotlib显示结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(o_rg, cv2.COLOR_BGR2RGB))
plt.title('RGB Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(o_de_masked, cmap='gray')  # 显示修改后的深度图
plt.title('Depth Image (Masked)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(o_i, cv2.COLOR_BGR2GRAY), cmap='gray')
plt.title('IR Image')
plt.axis('off')

plt.show()



