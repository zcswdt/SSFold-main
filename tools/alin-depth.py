import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

pipeline = rs.pipeline()

# 创建配置并配置要流式传输的管道
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

# 按照日期创建文件夹
save_path = os.path.join(os.getcwd(), "out", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, "color"), exist_ok=True)
os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)

# 保存的图片和实时的图片界面
cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)
saved_color_image = None  # 保存的临时图片
saved_depth_mapped_image = None
saved_count = 0

# 主循环
try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float16")
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("live", np.hstack((color_image, depth_mapped_image)))
        key = cv2.waitKey(30)

        # 按 's' 保存图片
        if key & 0xFF == ord('s'):
            saved_color_image = color_image
            saved_depth_mapped_image = depth_mapped_image

            # 彩色图片保存为png格式
            cv2.imwrite(os.path.join((save_path), "color", "{}.png".format(saved_count)), saved_color_image)
            # 深度信息由采集到的float16直接保存为npy格式
            np.save(os.path.join((save_path), "depth", "{}".format(saved_count)), depth_data)
            saved_count += 1
            cv2.imshow("save", np.hstack((saved_color_image, saved_depth_mapped_image)))

        # 按 'q' 退出
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()

#
# import pyrealsense2 as rs
# import numpy as np
# import cv2
#
# # 初始化Realsense管道
# pipeline = rs.pipeline()
# config = rs.config()
#
# # 启用RGB和深度流
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#
# # 开始流
# pipeline.start(config)
#
# try:
#     while True:
#         # 获取帧数据
#         frames = pipeline.wait_for_frames()
#
#         # 获取独立的深度帧和颜色帧
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#
#         # 确保帧数据是有效的
#         if not depth_frame or not color_frame:
#             continue
#
#         # 将图像转换为numpy数组
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
#
#         # 将深度图像转换为8位灰度图用于显示
#         depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
#
#         # 将深度图像转换为三通道灰度图
#         depth_colormap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
#
#         # 合并两个图像横向显示
#         images = np.hstack((color_image, depth_colormap))
#
#         # 显示图像
#         cv2.imshow('Unaligned Image', images)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     # 停止流
#     pipeline.stop()
#
# cv2.destroyAllWindows()
