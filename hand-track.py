import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 配置Realsense摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动Realsense摄像头
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()  # 等待获取帧
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # 将Realsense获取的帧转换为NumPy数组
        frame = np.asanyarray(color_frame.get_data())

        # 转换颜色空间为RGB用于手势识别
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)  # 处理图像获取手势识别结果

        # 在原始BGR图像上绘制手部关键点和连接线
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),  # 修改关键点的绘制样式
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2))  # 修改连接线的绘制样式

        # 显示图像
        cv2.imshow('Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 关闭Realsense摄像头
    pipeline.stop()
    cv2.destroyAllWindows()

