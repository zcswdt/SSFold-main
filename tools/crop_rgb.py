import os
import cv2
import numpy as np


def crop_image(image, keypoints, margin):
    # 获取裁剪区域的左上角和右下角坐标
    x_min = min(keypoints[:, 0]) + margin
    x_max = max(keypoints[:, 0]) - margin
    y_min = min(keypoints[:, 1]) + margin
    y_max = max(keypoints[:, 1]) - margin

    # 确保裁剪区域在图像边界内
    x_min = max(0, x_min)
    x_max = min(image.shape[1], x_max)
    y_min = max(0, y_min)
    y_max = min(image.shape[0], y_max)

    # 裁剪图像
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def process_images(input_dir, output_dir, keypoints, margin=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Error: Unable to open image {image_path}")
                    continue

                cropped_image = crop_image(image, keypoints, margin)
                output_path = os.path.join(output_dir, file)
                cv2.imwrite(output_path, cropped_image)
                print(f"Cropped image saved as {output_path}")




# 使用示例
input_dir = '/home/zcs/work/github/folding-by-hand/experiment/fold/SF/lv'
output_dir = '/home/zcs/work/github/folding-by-hand/experiment/fold/SF'
# rgb_keypoints = np.array([
#     [101, 83],
#     [422, 83],
#     [101, 410],
#     [422, 410]
# ])

rgb_keypoints = np.array([
    [90, 83],
    [411, 83],
    [90, 410],
    [411, 410]
])
# rgb_keypoints = np.array([
#     [101, 70],
#     [422, 70],
#     [101, 397],
#     [422, 397]
# ])
margin = 18

process_images(input_dir, output_dir, rgb_keypoints, margin)


