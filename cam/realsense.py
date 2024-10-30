import pyrealsense2 as rs
import numpy as np
import cv2
import os


class Camera:
    def __init__(self):
        self.pipeline = None
        self.queue = None
        self.W = 300  # Assuming the width is 300, adjust as needed

        # Initialize keypoints for homography
        self.ir_keypoints = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
        self.rgb_keypoints = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
        self.crop_keypoints = np.array([[0, 0], [self.W - 1, 0], [self.W - 1, self.W - 1], [0, self.W - 1]],
                                       dtype=np.float32)

        self.setup_camera()

    def get_intrinsics(self):
        # Get the camera intrinsics
        profile = self.pipeline.get_active_profile()
        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        # 将内参转换为3x3矩阵形式
        intrinsics = np.array(
            [color_intrinsics.fx, 0, color_intrinsics.ppx, 0, color_intrinsics.fy, color_intrinsics.ppy, 0, 0,
             1]).reshape(3, 3)

        return intrinsics

    def setup_camera(self):
        print("Setting Up Camera...")
        self.pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

        self.queue = rs.frame_queue(1)

        # Start streaming
        profile = self.pipeline.start(config, self.queue)
        self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(self.scale)
        depth_sensor = profile.get_device().first_depth_sensor()
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max) + 1):
            visual_preset = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
            print(f"{i}: {visual_preset}")
            if visual_preset == "Short Range":
                print("Found Preset")
                depth_sensor.set_option(rs.option.visual_preset, i)

        print("Camera Connected.")

    def get_image(self):
        frames = self.queue.wait_for_frame().as_frameset()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame()

        if not depth_frame or not color_frame or not ir_frame:
            print("Error: Could not retrieve frames.")
            return None, None, None, None

        depth = np.asanyarray(depth_frame.get_data())
        depth = depth * self.scale  # 将深度数据转换为实际单位

        color = np.asanyarray(color_frame.get_data())
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        ir = np.asanyarray(ir_frame.get_data())

        # ir_h, _ = cv2.findHomography(self.ir_keypoints, self.crop_keypoints)
        # rgb_h, _ = cv2.findHomography(self.rgb_keypoints, self.crop_keypoints)
        #
        # color_warped = cv2.warpPerspective(color, rgb_h, (self.W, self.W))
        # ir_warped = cv2.warpPerspective(ir, ir_h, (self.W, self.W))
        # depth_warped = cv2.warpPerspective(depth, ir_h, (self.W, self.W))
        #
        # color_rotated = cv2.rotate(color_warped, cv2.ROTATE_90_CLOCKWISE)
        # ir_rotated = cv2.rotate(ir_warped, cv2.ROTATE_90_CLOCKWISE)
        # depth_rotated = cv2.rotate(depth_warped, cv2.ROTATE_90_CLOCKWISE)
        #
        # mask = ir_rotated > 20
        # color_rotated[mask < 1] = (0, 0, 0)
        #return color_rotated, depth_rotated, ir_rotated, mask

        return color, depth, ir, ir

    def get_origin(self):
        frames = self.queue.wait_for_frame().as_frameset()

        # 创建对齐对象，将所有帧对齐到彩色帧
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)

        # 获取对齐后的帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        ir_frame = aligned_frames.first(rs.stream.infrared)

        if not depth_frame or not color_frame or not ir_frame:
            print("Error: Could not retrieve frames.")
            return None, None, None

        # 获取并处理深度数据
        depth = np.asanyarray(depth_frame.get_data())
        depth = depth * self.scale  # 将深度数据转换为实际单位
        # # fill 0 in depth with reasonable stuff
        depth = cv2.inpaint(
            depth.astype(np.float32),
            (depth==0).astype(np.uint8),
            inpaintRadius=0, flags=cv2.INPAINT_NS)

        # 获取并转换彩色数据
        color = np.asanyarray(color_frame.get_data())
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        # 获取红外数据
        ir = np.asanyarray(ir_frame.get_data())

        return color, depth, ir

    def save_images(self, color, depth, ir, mask, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(os.path.join(folder_path, 'color.png'), color)
        np.save(os.path.join(folder_path, 'depth_origin.npy'), depth)
        cv2.imwrite(os.path.join(folder_path, 'ir.png'), ir)
        cv2.imwrite(os.path.join(folder_path, 'mask.png'), mask.astype(np.uint8) * 255)  # 转换 mask 为 uint8 类型并放大

    def save_origin_images(self, color, depth, ir, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(os.path.join(folder_path, 'color_origin.png'), color)
        np.save(os.path.join(folder_path, 'depth_origin.npy'), depth)
        cv2.imwrite(os.path.join(folder_path, 'ir_origin.png'), ir)


if __name__ == "__main__":
    camera = Camera()
    color, depth, ir, mask = camera.get_image()
    cam_intr = camera.get_intrinsics()
    color_origin, depth_origin, ir_origin = camera.get_origin()

    if color is not None and depth is not None and ir is not None and mask is not None:
        folder_path = 'saved_images'
        camera.save_images(color, depth, ir, mask, folder_path)
        print(f"Processed images saved to {folder_path}.")
    else:
        print("Failed to retrieve processed images.")

    if color_origin is not None and depth_origin is not None and ir_origin is not None:
        folder_path_origin = 'saved_origin_images'
        camera.save_origin_images(color_origin, depth_origin, ir_origin, folder_path_origin)
        print(f"Original images saved to {folder_path_origin}.")
    else:
        print("Failed to retrieve original images.")






#
#
#
#
#
# import logging
# import matplotlib.pyplot as plt
# import numpy as np
# import pyrealsense2 as rs
#
# logger = logging.getLogger(__name__)
#
# class RealSenseCamera:
#     def __init__(self, device_id, width=640, height=480, fps=30):
#         self.device_id = device_id
#         self.width = width
#         self.height = height
#         self.fps = fps
#
#         self.pipeline = None
#         self.scale = None
#         self.intrinsics = None
#
#     def connect(self):
#         # Start and configure
#         self.pipeline = rs.pipeline()
#         config = rs.config()
#         config.enable_device(str(self.device_id))
#         config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
#         config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
#         config.enable_stream(rs.stream.infrared, self.width, self.height, rs.format.y8, self.fps)
#         cfg = self.pipeline.start(config)
#
#         # Determine intrinsics
#         rgb_profile = cfg.get_stream(rs.stream.color)
#         self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
#
#         # Determine depth scale
#         self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
#
#     def get_image_bundle(self):
#         frames = self.pipeline.wait_for_frames()
#
#         align = rs.align(rs.stream.color)
#         aligned_frames = align.process(frames)
#         color_frame = aligned_frames.first(rs.stream.color)
#         aligned_depth_frame = aligned_frames.get_depth_frame()
#         infrared_frame = aligned_frames.first(rs.stream.infrared)
#
#         depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
#         depth_image *= self.scale
#         color_image = np.asanyarray(color_frame.get_data())
#         infrared_image = np.asanyarray(infrared_frame.get_data())
#
#         depth_image = np.expand_dims(depth_image, axis=2)
#
#         return {
#             'rgb': color_image,
#             'aligned_depth': depth_image,
#             'infrared': infrared_image
#         }
#
#     def plot_image_bundle(self):
#         images = self.get_image_bundle()
#
#         rgb = images['rgb']
#         depth = images['aligned_depth']
#         infrared = images['infrared']
#
#         fig, ax = plt.subplots(1, 3, squeeze=False)
#         ax[0, 0].imshow(rgb)
#         m, s = np.nanmean(depth), np.nanstd(depth)
#         ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
#         ax[0, 2].imshow(infrared, cmap='gray')
#         ax[0, 0].set_title('RGB')
#         ax[0, 1].set_title('Aligned Depth')
#         ax[0, 2].set_title('Infrared')
#
#         plt.show()
#
# if __name__ == '__main__':
#     cam = RealSenseCamera(device_id='f1480545')
#     cam.connect()
#     while True:
#         cam.plot_image_bundle()







