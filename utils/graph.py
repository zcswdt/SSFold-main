from scipy import spatial
import numpy as np
import torch
#import pcl
import open3d as o3d
from utils.visual import get_world_coords_from_pixels
import cv2
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
def compute_edge_attr(normalized_vox_pc, neighbor_radius):
    point_tree = spatial.cKDTree(normalized_vox_pc)
    undirected_neighbors = np.array(list(point_tree.query_pairs(neighbor_radius, p=2))).T

    if len(undirected_neighbors) > 0:
        dist_vec = normalized_vox_pc[undirected_neighbors[0, :]] - normalized_vox_pc[undirected_neighbors[1, :]]
        dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
        edge_attr = np.concatenate([dist_vec, dist], axis=1)
        edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

        # Generate directed edge list and corresponding edge attributes
        edges = torch.from_numpy(np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1))
        edge_attr = torch.from_numpy(np.concatenate([edge_attr, edge_attr_reverse]))
    else:
        print("number of distance edges is 0! adding fake edges")
        edges = np.zeros((2, 2), dtype=np.uint8)
        edges[0][0] = 0
        edges[1][0] = 1
        edges[0][1] = 0
        edges[1][1] = 2
        edge_attr = np.zeros((2, 4), dtype=np.float32)
        edges = torch.from_numpy(edges).bool()
        edge_attr = torch.from_numpy(edge_attr)
        print("shape of edges: ", edges.shape)
        print("shape of edge_attr: ", edge_attr.shape)

    return edges, edge_attr


def build_graph(vox_pc, neighbor_radius):
    """
    Input:
    pointcloud

    Return:
    node_attr: N x (vel_history x 3)
    edges: 2 x E, the edges
    edge_attr: E x edge_feature_dim
    """
    normalized_vox_pc = vox_pc - np.mean(vox_pc, axis=0)
    node_attr = torch.from_numpy(normalized_vox_pc)
    edges, edge_attr = compute_edge_attr(normalized_vox_pc, neighbor_radius)

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # # 4. 可视化点云和边缘
    # node_attr_np = node_attr.numpy()
    #
    # # 提取 x, y, z 坐标
    # x = node_attr_np[:, 0]
    # y = node_attr_np[:, 1]
    # z = node_attr_np[:, 2]
    #
    # # 创建 3D 图形
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 绘制点云
    # ax.scatter(x, y, z, c=z, cmap='jet', marker='o', s=10)
    #
    # # 绘制每条边
    # for i in range(edges.shape[1]):
    #     start_idx, end_idx = edges[:, i]
    #     x_edge = [x[start_idx], x[end_idx]]
    #     y_edge = [y[start_idx], y[end_idx]]
    #     z_edge = [z[start_idx], z[end_idx]]
    #
    #     # 绘制边缘 (线段)
    #     ax.plot(x_edge, y_edge, z_edge, color='gray')
    #
    # # 设置 X、Y、Z 轴的范围
    # ax.set_xlim(-0.1, 0.1)
    # ax.set_ylim(-0.1, 0.1)
    # ax.set_zlim(-0.1, 0.1)
    #
    # # 保留坐标轴标签
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # # 显示点云和边缘
    # plt.show()



    # 可视化点云和边
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(normalized_vox_pc)
    # colors = np.tile([1, 0, 0], (normalized_vox_pc.shape[0], 1))  # 红色
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud], window_name="Sampled Point Cloud")

    # 创建线段
    lines = []
    for i in range(edges.shape[1]):
        lines.append([edges[0, i], edges[1, i]])

    # 创建 Open3D 线段集合对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(normalized_vox_pc)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_colors = [[0, 1, 0] for _ in range(len(lines))]  # 绿色
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    # 可视化点云和线段
    o3d.visualization.draw_geometries([point_cloud, line_set], window_name="Graph Visualization")

    # 创建点云
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(normalized_vox_pc)
    colors = np.tile([1, 0, 0], (normalized_vox_pc.shape[0], 1))  # 红色点云
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 创建线段集合
    lines = []
    for i in range(edges.shape[1]):
        lines.append([edges[0, i], edges[1, i]])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(normalized_vox_pc)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_colors = [[0, 1, 0] for _ in range(len(lines))]  # 绿色线段
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    #
    # # 使用 Visualizer 来保存图像
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="Point Cloud and Lines Visualization", width=800, height=600)
    # vis.get_render_option().background_color = np.array([0, 0, 0])  # 设置背景颜色为黑色
    # vis.add_geometry(point_cloud)
    # vis.add_geometry(line_set)
    # vis.update_geometry(point_cloud)
    # vis.update_geometry(line_set)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image("point_cloud_and_lines_black_background.png")
    # vis.destroy_window()

    return {"x": node_attr, "edge_index": edges, "edge_attr": edge_attr}


def voxelize_pointcloud(pointcloud, voxel_size):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pointcloud)
    downpcd = cloud.voxel_down_sample(voxel_size)  
    # cloud = pcl.PointCloud(pointcloud)
    # sor = cloud.make_voxel_grid_filter()
    # sor.set_leaf_size(voxel_size, voxel_size, voxel_size)   
    # pointcloud = sor.filter()
    points = np.asarray(downpcd.points).astype(np.float32)  #600个点左右

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # # 假设 point_cloud 是你的点云数据，形状为 (N, 3)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 提取 x, y, z 坐标
    # x = points[:, 0]
    # y = points[:, 1]
    # z = points[:, 2]
    #
    # # 绘制 3D 点云
    # ax.scatter(x, y, z, c=z, cmap='jet', marker='o')
    #
    # ax.set_xlim(-0.15, 0.15)
    # ax.set_ylim(-0.15, 0.15)
    # ax.set_zlim(0.75, 0.95)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # plt.show()


    # import matplotlib.pyplot as plt
    # # 使用 matplotlib 进行可视化
    # # 使用 matplotlib 进行可视化
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # ax.set_title("Voxelized Point Cloud")
    # plt.show()

    # # 使用 open3d 进行可视化
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([cloud])



    # # 创建一个点云对象
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(points)
    #
    # # 设置点云颜色，例如红色
    # colors = np.array([[1, 0, 0] for _ in range(len(points))])  # RGB, 红色
    # cloud.colors = o3d.utility.Vector3dVector(colors)
    #
    # # 创建一个可视化窗口
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    #
    # # 将点云添加到可视化窗口
    # vis.add_geometry(cloud)
    #
    # # 渲染窗口
    # vis.update_geometry(cloud)
    # vis.poll_events()
    # vis.update_renderer()
    #
    # # 保存当前视图到一个图像文件
    # vis.capture_screen_image("voxelize.png")
    #
    # # 关闭可视化窗口
    # vis.destroy_window()

    return points


def get_mask(depth):
    mask = depth.copy()
    mask[mask <0.01] = 0
    mask[mask != 0] = 1
    return mask


def adjust_extrinsics(extrinsics, h_rgb, scale):
    """
    Adjusts the extrinsics matrix based on the image transformation parameters.

    Args:
    - extrinsics: (4, 4) array, original camera extrinsics.
    - h_rgb: (3, 3) array, homography matrix used for image transformation.
    - scale: float, scaling factor applied after the transformation.

    Returns:
    - adjusted_extrinsics: (4, 4) array, adjusted camera extrinsics.
    """
    # Invert the homography matrix and convert to 4x4 by adding a row and a column
    homography_inv = np.linalg.inv(h_rgb)
    homography_inv_4x4 = np.eye(4)
    homography_inv_4x4[:3, :3] = homography_inv

    # Create a scaling matrix in 4x4
    scale_matrix = np.array([
        [1 / scale, 0, 0, 0],
        [0, 1 / scale, 0, 0],
        [0, 0, 1 / scale, 0],
        [0, 0, 0, 1]
    ])

    # Compute the adjusted extrinsics
    adjusted_extrinsics = extrinsics @ homography_inv_4x4 @ scale_matrix
    return adjusted_extrinsics

def plot_3d_cloud(sampled_pc, xlim=(-0.15, 0.15), ylim=(-0.15, 0.15), zlim=(0.6, 1), xlabel='X Label', ylabel='Y Label', zlabel='Z Label', save_path=None):
    """
    绘制3D点云的通用函数。

    输入:
    - sampled_pc: 点云数据 (N, 3)
    - xlim, ylim, zlim: X, Y, Z 轴的范围
    - xlabel, ylabel, zlabel: 轴的标签
    """
    # 创建一个新图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取 x, y, z 坐标
    x = sampled_pc[:, 0]
    y = sampled_pc[:, 1]
    z = sampled_pc[:, 2]

    # 绘制 3D 点云
    ax.scatter(x, y, z, c=z, cmap='jet', marker='o')

    # 设置 X、Y、Z 轴的范围
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # 设置轴标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    # 显示图形
    plt.show()
    # # 关闭当前图形窗口
    # plt.close(fig)
    if save_path:
        plt.savefig(save_path)
    # plt.close(fig)

def get_sampled_pc(depth, voxel_size, K, intrinsics,camera_pose,mask): #camera_params是相机标定的外参矩阵
    #mask = get_mask(depth)   #这个mask是获得图像中布料的区域，是在64*64的图像中求的
    #需要将depth从64*64-300*300，然后在到640*480，然后才能利用相机内参和相机坐标
    # 设置关键点
    crop_keypoints = np.array([
        [0, 0],
        [300, 0],
        [0, 300],
        [300, 300]
    ])

    rgb_keypoints = np.array([
        [117, 72],
        [444, 72],
        [117, 401],
        [443, 402]
    ])

    ir_keypoints = np.array([
        [167, 104],
        [411, 104],
        [168, 346],
        [409, 349]
    ])

    # 计算单应性矩阵
    h_rgb, _ = cv2.findHomography(rgb_keypoints, crop_keypoints)
    h_ir, _ = cv2.findHomography(ir_keypoints, crop_keypoints)
    # 计算逆单应性矩阵
    inv_h_ir = np.linalg.inv(h_ir)

    # 创建一个空的点云数组
    point_cloud = []

    depth = cv2.inpaint(
        depth.astype(np.float32),
        (depth == 0).astype(np.uint8),
        inpaintRadius=0, flags=cv2.INPAINT_NS)
    # 获取缩放后图像的尺寸
    height, width = depth.shape

    # 迭代每个像素
    for v in range(height):
        for u in range(width):
            # 获取深度值
            z = depth[v, u]

            if z > 0:  # 确保深度值有效
                # 逆变换到原始图像坐标
                u_rescaled = u * (300 / 64)
                v_rescaled = v * (300 / 64)
                src_pt = np.array([[[u_rescaled, v_rescaled]]], dtype='float32')
                src_pt = cv2.perspectiveTransform(src_pt, inv_h_ir)[0][0]
                u_orig, v_orig = src_pt

                # 计算相机坐标
                X_c = (u_orig - intrinsics[0, 2]) * z / intrinsics[0, 0]
                Y_c = (v_orig - intrinsics[1, 2]) * z / intrinsics[1, 1]
                Z_c = z

                # 相机坐标
                P_c = np.array([X_c, Y_c, Z_c, 1.0])

                # # 转换为世界坐标
                # P_w = np.dot(camera_pose, P_c)

                # 添加到点云
                point_cloud.append(P_c[:3])

    # 转换为numpy数组并调整维度
    point_cloud = np.array(point_cloud).reshape(-1, 3)
    pointcloud = point_cloud[mask.flatten() > 0].astype(np.float32)  # 741个点
    vox_pc = voxelize_pointcloud(pointcloud, voxel_size)  # 600个点左右 #voxel_size=0.0125
    sampled_pc = fps(vox_pc, K).astype(np.float32)  # 200个点

    # # 可视化 pointcloud
    # # 使用 Open3D 进行可视化
    sampled_cloud1 = o3d.geometry.PointCloud()
    sampled_cloud1.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.visualization.draw_geometries([sampled_cloud1], window_name="Sampled Point Cloud")

    # # #使用Plt绘制
    # plot_3d_cloud(pointcloud)
    # plot_3d_cloud(sampled_pc)

    # 可视化 sampled_pc
    # 使用 Open3D 进行可视化
    sampled_cloud = o3d.geometry.PointCloud()
    sampled_cloud.points = o3d.utility.Vector3dVector(sampled_pc)
    # 设置颜色
    # colors = np.tile([1, 0, 0], (sampled_pc.shape[0], 1))  # 红色
    # sampled_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([sampled_cloud], window_name="Sampled Point Cloud")



    return sampled_pc  #ndarray：（200，3）


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def fps(pts, K):   #FPS 是一种点云采样算法，旨在从原始点云中选取 K 个点，使得这些点在空间中的分布尽可能远离彼此，从而保持点云的代表性
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts
