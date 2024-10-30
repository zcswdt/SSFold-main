import threading
import numpy as np
from robots.geometry import transform_pose
from robots.ur5_robot import UR5RTDE
from robots.robotiq_gripper import RobotiqGripper
from cam.realsense import Camera
import cv2
import time
from scipy.spatial.transform import Rotation as R


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

ir_keypoints = np.array([
    [298, 102],
    [523, 100],
    [298, 325],
    [522, 325]
])




class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)

class SingleArmTableScene:
    """
    单臂桌面场景类，包含机器人和相机的初始化及操作方法。
    """

    def __init__(self, tx_table_camera, tx_camera, robot, confirm_actions=True):
        # 计算手臂到世界坐标的变换矩阵
        self.tx_world = tx_camera @ np.linalg.inv(tx_table_camera)
        self.robot = robot
        self.confirm_actions = confirm_actions
        self.f = 0
        # 打开机械手
        if not self.open_grippers(sleep_time=0, blocking=False):
            raise Exception("无法打开机械手")

    def disconnect(self):
        self.robot.disconnect()

    def single_arm_movel(self, p, speed=0.25, acceleration=1.2, blocking=True, avoid_singularity=False):
        tx = self.tx_world
        robot = self.robot
        rp = transform_pose(tx, p)
        return robot.movel(rp, speed, acceleration, blocking, avoid_singularity=avoid_singularity)


    def open_grippers(self, sleep_time=1, blocking=True):
        t1 = ThreadWithResult(target=self.robot.open_gripper, args=(0,))
        t1.start()
        if blocking:
            t1.join()
            time.sleep(sleep_time)
            return t1.result
        return True

    def close_grippers(self, sleep_time=1, blocking=True):
        t1 = ThreadWithResult(target=self.robot.close_gripper, args=(1,))
        t1.start()
        if blocking:
            t1.join()
            time.sleep(sleep_time)
            return t1.result
        return True

    def home(self, speed=1.5, acceleration=1, blocking=True):
        t1 = ThreadWithResult(target=self.robot.home, args=(speed, acceleration, blocking))
        t1.start()
        if blocking:
            t1.join()
            return t1.result
        return True

    def compute_pose(self,pick_point, place_point):
        # 计算方向向量并标准化，作为新的X轴
        direction_vector = np.array(place_point) - np.array(pick_point)
        x_axis = direction_vector / np.linalg.norm(direction_vector)

        # Z轴向下
        z_axis = np.array([0, 0, -1])

        # 计算Y轴，确保是与X轴和Z轴正交的
        y_axis = np.cross(z_axis, x_axis)
        y_axis_normalized = y_axis / np.linalg.norm(y_axis)

        # 重新计算X轴以确保坐标系正交性
        x_axis = np.cross(y_axis_normalized, z_axis)

        # 创建旋转矩阵
        rotation_matrix = np.column_stack((x_axis, y_axis_normalized, z_axis))
        # 从旋转矩阵转换为旋转向量
        rotation = R.from_matrix(rotation_matrix)
        rotation_vector = rotation.as_rotvec()


        pose = list(pick_point) + list(rotation_vector)
        return pose

    def adjust_place_position(self,pick, place, d):
        # 计算pick到place的向量
        vector = np.array(place) - np.array(pick)

        # 只考虑x和y分量
        x, y = vector[0], vector[1]

        # 计算方向角（以弧度为单位）
        angle = np.arctan2(y, x)

        # 根据角度判断所在象限并进行微调
        if 0 <= angle < np.pi / 2:
            # 第一象限
            new_place = np.array([place[0] - d, place[1] - d, place[2]])
        elif np.pi / 2 <= angle < np.pi:
            # 第二象限
            new_place = np.array([place[0] + d, place[1] - d, place[2]])
        elif -np.pi <= angle < -np.pi / 2:
            # 第三象限
            new_place = np.array([place[0] + d, place[1] + d, place[2]])
        elif -np.pi / 2 <= angle < 0:
            # 第四象限
            new_place = np.array([place[0] - d, place[1] + d, place[2]])
        else:
            # 处理角度不在预期范围内的情况（虽然不应该发生）
            new_place = np.array(place)

        return new_place

    def adjust_place(self, pick, place, d):
        # 将 place 和 pick 转换为 numpy 数组
        place = np.array(place)
        pick = np.array(pick)

        # 计算方向向量
        direction_vector = pick - place
        # 计算方向向量的模
        distance = np.linalg.norm(direction_vector)

       # 标准化方向向量
        normalized_direction = direction_vector / distance
        #计算新的 place 坐标
        new_place = place + d * normalized_direction
        return new_place


    def single_arm_pick_and_place(self, start, end,min_pick_height=0.036, lift_height=0.06, speed=0.3, acceleration=0.5):
        #min_pick_height = 0.037,0.038
        dist = np.linalg.norm(end - start)
        print('start',start)
        print('end', end)
        if dist > 0.25 and self.f<1:
            end = self.adjust_place(start, end, 0.03)
            self.f += 1
            # 计算抓取和放置的位姿lift_height=0.07
            start_pose = np.array([0, 0, 0, 0, np.pi, 0])
            end_pose = start_pose.copy()
            start_pose[:2], start_pose[2] = start[:2], max(start[2], min_pick_height)
            end_pose[:2], end_pose[2] = end[:2], max(end[2], min_pick_height)
            start_lift_pose, end_lift_pose = start_pose.copy(), end_pose.copy()
            start_lift_pose[2] = lift_height
            end_lift_pose[2] = lift_height
            print('fold1')
        elif dist > 0.25 and self.f==1:
            end = self.adjust_place(start, end, 0.03)
            # 计算抓取和放置的位姿lift_height=0.07
            start_pose = np.array([0, 0, 0, 0, np.pi, 0])
            end_pose = start_pose.copy()
            start_pose[:2], start_pose[2] = start[:2], max(start[2], 0.038)
            end_pose[:2], end_pose[2] = end[:2], max(end[2], 0.04)
            start_lift_pose, end_lift_pose = start_pose.copy(), end_pose.copy()
            start_lift_pose[2] = lift_height
            end_lift_pose[2] = lift_height
            print('fold2')
        end = self.adjust_place(start, end, 0.015)
        start_pose = np.array([0, 0, 0, 0, np.pi, 0])
        end_pose = start_pose.copy()
        start_pose[:2], start_pose[2] = start[:2], max(start[2], min_pick_height)
        end_pose[:2], end_pose[2] = end[:2], max(end[2], 0.045)
        start_lift_pose, end_lift_pose = start_pose.copy(), end_pose.copy()
        start_lift_pose[2] = lift_height
        end_lift_pose[2] = lift_height
        robot = self.robot

        tcp_pose = robot.get_current_joint_positions()
        print("当前关节角度:", tcp_pose)
        # 打开机械手
        robot.open_gripper(0)
        #移动到预抓取位置
        r = self.single_arm_movel(
            p=np.array([start_lift_pose]),
            speed=speed, acceleration=acceleration,
            avoid_singularity=True)
        if not r: return False

        #旋转夹爪
        pick_point = start
        place_point = end
        pick_point[2] = 0.07
        place_point[2] = 0.07
        pose = self.compute_pose(pick_point, place_point)
        r = self.single_arm_movel(
            p=np.array([pose]),
            speed=2, acceleration=2,
            avoid_singularity=True)
        if not r: return r

        # 移动到抓取位置
        pick_pose = start_pose
        pick_pose[3:] = pose[3:]
        r = self.single_arm_movel(
            p=np.array([pick_pose]),
            speed=speed, acceleration=acceleration,
            avoid_singularity=True)
        if not r: return r
        robot.close_gripper(1)

        start_lift_pose[3:] = pose[3:]
        end_lift_pose[3:] = pose[3:]
        r = self.single_arm_movel(
            p=np.array([start_lift_pose, end_lift_pose]),
            speed=speed, acceleration=acceleration)

        #移动到放置位置，
        end_pose[3:] = pose[3:]
        r = self.single_arm_movel(
            p=np.array([end_pose]),
            speed=speed, acceleration=acceleration)
        if not r: return r
        robot.open_gripper(1)

        #移动到预抓取位置
        r = self.single_arm_movel(
            p=end_lift_pose,
            speed=speed, acceleration=acceleration)

        return r

    def pick_and_place(self, start_point, end_point):

        r = self.single_arm_pick_and_place(start_point, end_point)
        if not r : return False
        return self.home()



class RobotSystem:
    def __init__(self):
        # 从文件中加载相机到桌面的变换矩阵和相机本身的配置
        self.tx_table_camera = np.loadtxt(
            '/home/zcs/work/github/ssfold/calibration/cam2table_pose_2.txt')
        self.tx_camera = np.loadtxt(
            '/home/zcs/work/github/ssfold/calibration/cam_pose.txt')

        # 初始化机械手和连接设置
        self.gripper = RobotiqGripper()
        self.gripper.connect('192.168.8.100', 63352)

        # 初始化UR5机械臂并关联机械手
        self.ur5 = UR5RTDE('192.168.8.100', self.gripper)
        self.cam = Camera()
        self.cam_intr = self.cam.get_intrinsics()
        # 创建机械臂操作场景，设置初始位置
        self.scene = SingleArmTableScene(tx_table_camera=self.tx_table_camera, tx_camera=self.tx_camera, robot=self.ur5)
        self.f = 0

    def init_robots(self):
        # 激活并重置机械手，输出状态信息
        print("Activating gripper...")
        self.gripper._reset()
        self.gripper.activate()
        print('Robot ready')
        # 机械臂移动到初始位置
        self.scene.home(speed=0.5)

    def find_nearest_nonzero(self, pos, mask):
        # 在给定的mask中找到最接近指定位置的非零元素
        pos = np.array(pos)
        inds = np.argwhere(mask)
        dists = np.linalg.norm(inds - pos, axis=1)
        return inds[np.argmin(dists)]

    def map_crop_to_rgb(self, input, rgb_keypoints, crop_keypoints):
        # 通过计算单应性矩阵将裁剪图像的坐标映射到原始图像坐标
        u, v = input
        rgb_h, _ = cv2.findHomography(rgb_keypoints, crop_keypoints)
        inverse_rgb_h = np.linalg.inv(rgb_h)
        point = np.array([u, v, 1]).reshape(3, 1)
        transformed_point = np.dot(inverse_rgb_h, point)
        transformed_point /= transformed_point[2]  # 保证这里进行了归一化
        return transformed_point[0][0], transformed_point[1][0]

    # def pixel_to_world(self, u, v, M, tx_table_camera, intrinsics):
    #     # 将像素坐标转换为世界坐标
    #     x = (u - intrinsics[0, 2]) / intrinsics[0, 0]
    #     y = (v - intrinsics[1, 2]) / intrinsics[1, 1]
    #     uv_homogeneous = np.array([x, y, 1])
    #     world_coordinates = np.dot(M, np.append(uv_homogeneous, [1]))
    #     return world_coordinates[:3]


    def pixel_to_world(self, u, v, z, tx_table_camera, intrinsics):
        # 将像素坐标转换为世界坐标
        # x = (u - intrinsics[0, 2]) / intrinsics[0, 0]
        # y = (v - intrinsics[1, 2]) / intrinsics[1, 1]
        x = (u - intrinsics[0, 2]) / intrinsics[0, 0] * z
        y = (v - intrinsics[1, 2]) / intrinsics[1, 1] * z
        camera_coords = np.array([x, y, z, 1])

        # 将相机坐标系中的坐标转换为世界坐标系中的坐标
        world_coords = np.dot(tx_table_camera, camera_coords)

        # 返回世界坐标的前三个维度 (x, y, z)
        return world_coords[:3]

    def perform_action(self, pick, place, rgb_keypoints, crop_keypoints, depth, mask):  #pick，place，mask在64*64的图像，depth，640*480
        # 执行抓取和放置动作，包括坐标变换和位置计算
        # if not mask[int(pick[0]), int(pick[1])]:
        #     pick = self.find_nearest_nonzero(pick, mask)
        #     print("New Pick", pick)


        #pick = pick[::-1]
        #place = place[::-1]

        x, y = self.map_crop_to_rgb(pick, rgb_keypoints, crop_keypoints)  #将pick和place转成原始图像上的点
        x, y = round(y), round(x)
        z1 = depth[y, x]

        ex, ey = self.map_crop_to_rgb(place, rgb_keypoints, crop_keypoints)
        ex, ey = round(ey), round(ex)
        z2 = depth[ey, ex]

        # pick_world_coords = self.pixel_to_world(262, 238, z1, self.tx_table_camera, self.cam_intr)  # 将像素坐标转成世界坐标
        # place_world_coords = self.pixel_to_world(394, 338, z2, self.tx_table_camera, self.cam_intr)

        #三维坐标，包含深度信息
        pick_world_coords = self.pixel_to_world(y, x, z1, self.tx_table_camera, self.cam_intr)   #将像素坐标转成世界坐标
        place_world_coords = self.pixel_to_world(ey, ex, z2, self.tx_table_camera, self.cam_intr)



        # depth_filtered = depth * mask   #mask原图的
        #
        #
        # table_to_cam = 0.83
        # mean_point_meters = np.mean(depth_filtered[depth_filtered > 0]) * 0.00025
        # highest_point_meters = np.min(depth_filtered[depth_filtered > 0]) * 0.00025
        # max_height_of_cloth = table_to_cam - highest_point_meters
        # mean_height_of_cloth = table_to_cam - mean_point_meters
        #
        # pre_fold_height = 0.01 + max_height_of_cloth
        #
        # # 将最后一个坐标替换为 pre_fold_height
        # pick_world_coords[-1] = pre_fold_height
        # place_world_coords[-1] = pre_fold_height

        start_point = pick_world_coords
        end_point = place_world_coords


        # start_point = np.array([0,0,-0.02])
        # end_point = np.array([0.08,0.08,-0.02])
        start_point[1] -= 0.013  # 调整偏差
        start_point[0] -= 0.003

        end_point[1] -= 0.013  # 调整偏差
        end_point[0] -= 0.003

        self.scene.pick_and_place(start_point, end_point)




if __name__ == '__main__':
    # Usage
    camera = Camera()
    color, depth, ir, mask = camera.get_image()
    robot_system = RobotSystem()


    indices = np.transpose(np.nonzero(color))  # 获取彩色图像中非零像素点的坐标索引
    ind = indices[np.random.randint(indices.shape[0])]  # 随机选择一个非零像素点的索引
    end = np.random.randint(0, 300, 2)  # 随机生成一个目标位置

    color = cv2.arrowedLine(color, (ind[1], ind[0]), (end[1], end[0]), (0, 250, 100), 2)  # 在彩色图像上绘制一条从选定点到目标点的箭头
    color = cv2.circle(color, (ind[1], ind[0]), 5, (0, 0, 250), 2)
    color = cv2.circle(color, (end[1], end[0]), 5, (250, 0, 0), 2)
    cv2.destroyAllWindows()
    cv2.imshow("color", color)
    cv2.waitKey(1)

    robot_system.perform_action(ind, end, rgb_keypoints, crop_keypoints, depth, mask)




