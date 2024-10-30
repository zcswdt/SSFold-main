from pyk4a import PyK4A, Config, ColorResolution, FPS
import cv2
import time


def save_rgb_video(output_file, fps=30):
    # 设置相机配置
    config = Config(
        color_resolution=ColorResolution.RES_720P,  # 或其他你需要的分辨率
        camera_fps=FPS.FPS_30,  # 设置帧率
        synchronized_images_only=True,  # 确保颜色和深度帧同步
    )

    # 初始化相机
    k4a = PyK4A(config=config)
    k4a.start()

    # 创建视频写入器，这里假定使用720P分辨率，你可以根据实际分辨率调整
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (1280, 720))  # 根据所选分辨率调整

    # 创建窗口
    cv2.namedWindow('RGB', cv2.WINDOW_AUTOSIZE)

    # 捕获视频
    recording = True
    while recording:
        capture = k4a.get_capture()
        if capture.color is not None:
            img = capture.color[:, :, :3]  # 获取 RGB 图像
            out.write(img)  # 写入视频帧
            cv2.imshow('RGB', img)  # 显示图像

            # 检测窗口是否关闭
            if cv2.getWindowProperty('RGB', cv2.WND_PROP_VISIBLE) < 1:
                recording = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 释放资源
    out.release()
    cv2.destroyAllWindows()
    k4a.stop()


# 使用示例
save_rgb_video('output.avi', fps=30)




#
# import pickle
# import requests
# import cv2
# import numpy as np
# import time
#
#
# class KinectClient:
#     def __init__(self, ip, port, video_file='output.avi', fps=30):
#         self.ip = ip
#         self.port = port
#         self.video_file = video_file
#         self.fps = fps
#         self.out = None  # 视频写入器对象
#
#     def get_intr(self):
#         return pickle.loads(requests.get(f'http://{self.ip}:{self.port}/intr').content)
#
#     def start_video_recording(self, frame_size):
#         # 初始化视频写入器
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         self.out = cv2.VideoWriter(self.video_file, fourcc, self.fps, frame_size)
#         print(f"Video recording started, saving to {self.video_file}")
#
#     def stop_video_recording(self):
#         # 释放视频写入器
#         if self.out is not None:
#             self.out.release()
#             print("Video recording stopped and file closed.")
#         else:
#             print("No video recording was started.")
#
#     def get_camera_data(self, n=1):
#         cam_data = pickle.loads(requests.get(f'http://{self.ip}:{self.port}/pickle/{n}').content)
#         color_img = cam_data['color_img']
#
#         # 确保图像数据是 uint8 类型
#         if color_img.dtype != np.uint8:
#             color_img = color_img.astype(np.uint8)
#         # 转换 RGB 到 BGR
#         color_img_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
#         # 确保图像尺寸与视频写入器的 frame_size 匹配
#         #print(f"Image shape: {color_img.shape}")
#
#         # 写入视频帧
#         if self.out is not None:
#             self.out.write(color_img_bgr)
#         else:
#             print("Video writer not initialized.")
#
#         return color_img_bgr
#
#
# # 使用示例
# kinect = KinectClient(ip='192.168.8.101', port=8080, video_file='output.avi', fps=30)
# frame_size = (1280, 720)  # 根据实际的 RGB 图像尺寸调整
# kinect.start_video_recording(frame_size=frame_size)
#
# # 假设我们要录制20秒的视频
# start_time = time.time()
# duration = 20000
# while time.time() - start_time < duration:
#     color_img = kinect.get_camera_data()
#     # 在这里可以显示图像或进行其他处理
#     # cv2.imshow('RGB', color_img)  # 如果需要调试显示图像
#     # cv2.waitKey(1)
#
# kinect.stop_video_recording()

