import numpy as np  # 导入numpy库，用于数组和矩阵运算
import pyrealsense2 as rs  # 导入pyrealsense2库，用于访问RealSense摄像头的API
import cv2  # 导入OpenCV库，用于图像处理
from datetime import datetime
class Realsense(object):  # 定义Realsense类

    def __init__(self,width=640,height=480,fps=15,serial_number=None):  # 类的初始化函数，设置默认的图像宽度、高度和帧率
        self.serial_number = serial_number  # 添加序列号属性
        self.im_height = height  # 设置图像高度
        self.im_width = width  # 设置图像宽度
        self.fps = fps  # 设置帧率
        self.intrinsics = None  # 初始化内参变量
        self.scale = None  # 初始化深度比例变量
        self.pipeline = None  # 初始化pipeline变量
        self.connect()  # 调用connect函数，连接摄像头并进行配置

    def connect(self):  # 定义连接摄像头的函数
        self.pipeline = rs.pipeline()  # 创建一个pipeline
        config = rs.config()  # 创建一个配置对象
        if self.serial_number:
            config.enable_device(self.serial_number)  # 使用序列号连接到特定的相机
        # 为深度和彩色流配置参数
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)

        cfg = self.pipeline.start(config)  # 启动pipeline

        # 获取并设置RGB相机的内参
        self.rgb_profile = cfg.get_stream(rs.stream.color)
        #self.intrinsics = self.get_intr(self.rgb_profile)
        # 获取并设置深度比例
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        print("camera depth scale:",self.scale)  # 打印深度比例
        print("D415 have connected ...")  # 打印摄像头连接成功的信息

    def get_camera_data(self):  # 定义获取数据的函数

        frames = self.pipeline.wait_for_frames()  # 等待获取一帧数据

        # 对齐深度和彩色图像
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # # 将图像数据转换为numpy数组
        # depth_image = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.float32)
        # depth_image = np.expand_dims(depth_image, axis=2)  # 为深度图像增加一个维度
        # color_image = np.asanyarray(color_frame.get_data())

        # 将图像数据转换为numpy数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image = depth_image / 1000.0  # 将深度值从毫米转换为米
        depth_image = np.expand_dims(depth_image, axis=2)  # 为深度图像增加一个维度
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image  # 返回彩色图像和深度图像



    def get_rgb(self):  # 定义获取数据的函数
        import time
        time.sleep(0.01)
        frames = self.pipeline.wait_for_frames()  # 等待获取一帧数据

        # 对齐深度和彩色图像
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        return color_image



    def plot_image(self):  # 定义显示图像的函数
        color_image, depth_image = self.get_camera_data()  # 获取彩色和深度图像
        # 对深度图像应用颜色映射以便于显示
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 如果深度和彩色图像的分辨率不同，调整彩色图像的大小以匹配深度图像
        if depth_colormap.shape != color_image.shape:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap.shape[1], depth_colormap.shape[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # 显示图像
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(5000)  # 等待5秒

    def get_intr(self):  # 那定义获取相机内参的函数
        raw_intrinsics = self.rgb_profile.as_video_stream_profile().get_intrinsics()  # 获取内参
        # 将内参转换为3x3矩阵形式
        intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)
        print('intrinsics', intrinsics)  # 打印内参
        return intrinsics  # 返回内参矩阵

if __name__== '__main__':  # 当文件被直接运行时
    # context = rs.context()
    # # 获取连接的设备列表
    # devices = context.query_devices()
    # print("Found {} RealSense devices:".format(len(devices)))
    # for i, device in enumerate(devices):
    #     # 获取设备的序列号
    #     serial_number = device.get_info(rs.camera_info.serial_number)
    #     # 获取设备的型号
    #     model = device.get_info(rs.camera_info.name)
    #     print("Device {}: {}, Serial Number: {}".format(i + 1, model, serial_number))
    # print("Device {}: {}, Serial Number: {}".format(i + 1, model, serial_number))
    # 使用序列号创建两个Realsense实例
    # mycamera1 = Realsense(serial_number='044422250492')
    # mycamera2 = Realsense(serial_number='935422070190')
    #
    # mycamera1.plot_image()  # 调用plot_image函数显示图像
    # mycamera2.plot_image()
    # 创建两个Realsense实例，每个实例对应一个相机
    mycamera1 = Realsense(serial_number='044422250492')
    mycamera2 = Realsense(serial_number='935422070190')
    # 获取当前日期和时间作为字符串
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 定义两个输出视频文件的名称
    output_filename1 = f'/home/zcs/work/github/2024/cloth-funnels/real-log/realsense_camera1_{timestamp}.avi'
    output_filename2 = f'/home/zcs/work/github/2024/cloth-funnels/real-log/realsense_camera2_{timestamp}.avi'

    # 设置视频编码格式
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = 15  # 设置视频的帧率
    frame_size = (640, 480)  # 设置视频帧的大小，确保和相机配置的分辨率一致

    # 为每个相机创建一个VideoWriter实例
    out1 = cv2.VideoWriter(output_filename1, fourcc, fps, frame_size)
    out2 = cv2.VideoWriter(output_filename2, fourcc, fps, frame_size)

    try:
        print("Starting video capture. Press 'q' to exit...")
        while True:
            # 从每个相机捕获帧
            color_image1,_ = mycamera1.get_camera_data()
            color_image2,_ = mycamera2.get_camera_data()

            # 将帧写入各自的视频文件
            out1.write(color_image1)
            out2.write(color_image2)

            # 显示视频帧
            cv2.imshow('Camera 1 Frame', color_image1)
            cv2.imshow('Camera 2 Frame', color_image2)

            # 按'q'退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 释放VideoWriter资源
        out1.release()
        out2.release()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        print("Video capture ended.")