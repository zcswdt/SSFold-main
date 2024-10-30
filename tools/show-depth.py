import numpy as np
import matplotlib.pyplot as plt
import cv2

# 指定文件路径
file_path = "/home/zcs/work/github/folding-by-hand/pic/3_depth.npy"

# 加载深度图数据
depth_data = np.load(file_path)
depth_data = cv2.inpaint(
    depth_data.astype(np.float32),
    (depth_data == 0).astype(np.uint8),
    inpaintRadius=0, flags=cv2.INPAINT_NS)
# 直接保存深度图为图像文件，不显示色条或标题
output_path = "o_visualization11.png"
plt.imsave(output_path, depth_data, cmap='gray')


