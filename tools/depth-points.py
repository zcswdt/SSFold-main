import cv2
import torch
import glob
import json
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.data import Data
from scipy import spatial
import numpy as np
import torch
#import pcl
import open3d as o3d
from utils.visual import get_world_coords_from_pixels
import cv2


crop_keypoints = np.array([
    [0, 0],
    [300, 0],
    [0, 300],
    [300, 300],
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




def get_mask(ir):
    # set img to 0 where ir < 0.5
    ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
    mask = ir > 20
    return mask

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

    # 可视化点云和边
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(normalized_vox_pc)
    colors = np.tile([1, 0, 0], (normalized_vox_pc.shape[0], 1))  # 红色
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

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
    #
    # 使用 open3d 进行可视化
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([cloud])


    return points


# def get_mask(depth):
#     mask = depth.copy()
#     mask[mask <0.01] = 0
#     mask[mask != 0] = 1
#     return mask



def get_sampled_pc(depth, voxel_size, K, intrinsics,camera_pose,mask): #camera_params是相机标定的外参矩阵

    # 计算逆单应性矩阵
    inv_h_ir = np.linalg.inv(h_ir)

    # 创建一个空的点云数组
    point_cloud = []

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


    pointcloud = point_cloud[mask.flatten() > 0].astype(np.float32)  #741个点
    vox_pc = voxelize_pointcloud(pointcloud, voxel_size)   #600个点左右 #voxel_size=0.0125
    sampled_pc = fps(vox_pc, K).astype(np.float32)  #200个点

    # # 可视化 sampled_pc
    # # 使用 Open3D 进行可视化
    # sampled_cloud = o3d.geometry.PointCloud()
    # sampled_cloud.points = o3d.utility.Vector3dVector(sampled_pc)
    #
    # # 设置颜色
    # colors = np.tile([1, 0, 0], (sampled_pc.shape[0], 1))  # 红色
    # sampled_cloud.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([sampled_cloud], window_name="Sampled Point Cloud")





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


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--datasets", type=str, required=False, help="Paths to process", nargs='+',default=["data/red_triangle_val"])
    # parser.add_argument("--output_name", type=str, required=False, help="Name of output dataset",default="red_triangle_val")
    parser.add_argument("--datasets", type=str, required=False, help="Paths to process", nargs='+',
                        default=["data_my/train"])
    parser.add_argument("--output_name", type=str, required=False, help="Name of output dataset", default="test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # with open("../{}/homograph_data.json".format(args.datasets[0]), "r") as f:
    #     homograph_data = json.load(f)
    #     crop_keypoints = np.array(homograph_data["crop"])
    #     ir_keypoints = np.array(homograph_data["ir"])
    #     rgb_keypoints = np.array(homograph_data["rgb"])

    h_ir, _ = cv2.findHomography(ir_keypoints, crop_keypoints)
    h_rgb, _ = cv2.findHomography(rgb_keypoints, crop_keypoints)

    buf = []

    dataset_paths = []
    for dataset in args.datasets:
        dataset_paths += glob.glob("../{}/*".format(dataset))

    for episode_path in tqdm(dataset_paths):
        timestep = 0
        next_timestep_depth_path = "{}/{}_depth.npy".format(episode_path, timestep + 1)
        while next_timestep_depth_path in glob.glob("{}/*_depth.npy".format(episode_path)):
            try:
                o_file = "{}/{}_".format(episode_path, timestep)

                o_rgb = cv2.imread(o_file + "rgb.png")
                o_ir = cv2.imread(o_file + "ir.png")
                o_dep = np.load(o_file + "depth.npy")

                # # 将大于等于1的值设为1，小于1的值设为0
                # o_dep[o_dep >= 1] = 1
                # o_dep[o_dep < 1] = 0
                #
                # # 保存结果为图片
                # cv2.imwrite("output_image.png", o_dep * 255)

                o_ir = cv2.warpPerspective(o_ir, h_ir, (300, 300))
                o_dep = cv2.warpPerspective(o_dep, h_ir, (300, 300))
                o_rgb = cv2.warpPerspective(o_rgb, h_rgb, (300, 300))

                # 使用红外光来获得mask
                o_mask = get_mask(o_ir)
                # min_depth, max_depth = get_min_max(dataset_paths)
                # norm_o_dep = normalize_depth(o_dep, o_mask)

                # 缩放深度图
                o_de = cv2.resize(o_dep, (64, 64))

                # 对布尔o_mask进行resize
                o_mask_resized = cv2.resize(o_mask.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)

                # 将uint8类型的结果转换回布尔类型
                o_mask = o_mask_resized.astype(bool)

                # # 将布尔数组转换为整数数组
                # image_array = np.where(o_mask, 0, 255).astype(np.uint8)
                #
                # # 保存为图像
                # cv2.imwrite("o_mask.png", image_array)

                intrinsics = np.array([[604.99621582, 0.0, 329.07382202],
                                       [0.0, 605.07873535, 243.38549805],
                                       [0.0, 0.0, 1.0]])
                camera_pose = '/home/zcs/work/github/folding-by-hand/calibration/cam2table_pose.txt'
                camera_pose = np.loadtxt(camera_pose)

                # 需要将相机内参，相机外参传进去，求解从像素坐标-世界坐标的转换camera_params：相机内参
                sampled_pc = get_sampled_pc(depth=o_de, voxel_size=0.0125, K=200, intrinsics=intrinsics, camera_pose=camera_pose, mask =o_mask)

                graph_data = build_graph(sampled_pc, 0.025)
                graph_data = Data.from_dict(graph_data)




                timestep += 1
                next_timestep_depth_path = "{}/{}_depth.npy".format(episode_path, timestep + 1)
            except Exception as e:
                print(e)
                break



