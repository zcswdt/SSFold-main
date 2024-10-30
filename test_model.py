from utils.model_utils import *
import utils.visualize as viz

from models.picktoplace import PickToPlaceModel
from models.pickandplace import PickAndPlaceModel

import torch
import collections

import random
import torch.nn as nn
import argparse
from tqdm import tqdm
import jsonlines

from utils.graph import build_graph, get_sampled_pc, calc_distances
from torch_geometric.data import Data, Batch


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights", type=str, required=False, help="weights filename",default="/home/zcs/work/github/ssfold/weights/green_triangle_y4hw0kuk_pick_and_place26500.pt")
  parser.add_argument("--use_depth", type=bool, default=True, help="Use depth for observations")
  parser.add_argument("--image_width", type=int, default=64, help="Image width")
  parser.add_argument("--heatmap_sigma", type=int, default=3, help="Sigma for gaussian heatmaps")

  return parser.parse_args()


def process_single_transition(transition, args):
    batch_elem = {}

    if args.use_depth:
        obs = transition['obs']['depth']
        obs = torch.FloatTensor(obs).cuda()
        obs = obs.unsqueeze(2)
    else:
        obs = transition['obs']['rgb']/255.0
        obs = torch.FloatTensor(obs).cuda()

    o_mask = transition['obs']['mask']
    o_mask_tensor = torch.from_numpy(o_mask).unsqueeze(2).cuda()  # 转换后的形状为 [64, 64, 1]
    obs_masked = obs * o_mask_tensor
    obs_mask = obs_masked.permute(2, 0, 1)

    pick = transition['pick']
    place_map = gaussian_heatmap(transition['place'], args.image_width, args.heatmap_sigma)
    pick_map = gaussian_heatmap(pick, args.image_width, args.heatmap_sigma)
    pick_map = torch.FloatTensor(pick_map).cuda().unsqueeze(0)
    place_map = torch.FloatTensor(place_map).cuda().unsqueeze(0)

    depth = transition['obs']['depth']
    intrinsics = np.array([[604.99621582, 0.0, 329.07382202],
                           [0.0, 605.07873535, 243.38549805],
                           [0.0, 0.0, 1.0]])
    camera_pose = '/home/zcs/work/github/folding-by-hand/calibration/cam2table_pose.txt'
    camera_pose = np.loadtxt(camera_pose)

    sampled_pc = get_sampled_pc(
        depth=depth, voxel_size=0.0125, K=200, intrinsics=intrinsics, camera_pose=camera_pose, mask=o_mask)

    graph_data = build_graph(sampled_pc, 0.025)
    graph_data = Data.from_dict(graph_data)

    batch_elem['obs'] = obs_mask
    batch_elem['pick'] = pick_map
    batch_elem['place'] = place_map
    batch_elem['place_index'] = transition['place']
    batch_elem['pick_index'] = transition['pick']
    batch_elem['graph'] = graph_data
    batch_elem['mask'] = transition['obs']['mask']

    return batch_elem

def transition_to_tensors(transition_elem):
    tensor_dict = {}

    # 直接处理每个键值对
    for key, value in transition_elem.items():
        if key in ["obs_rgb", "nobs_rgb", "pick_index", "place_index", "mask"]:
            # 这些不需要转换为张量
            tensor_dict[key] = value
        elif key == "graph":
            # 图数据的特殊处理
            tensor_dict[key] = Data(
                x=value.x,
                edge_index=value.edge_index,
                edge_attr=value.edge_attr,
                batch=torch.zeros(value.x.size(0), dtype=torch.long),
                ptr=torch.tensor([0, value.x.size(0)], dtype=torch.long)
            )
        else:
            # 其他都转换为单个张量
            tensor_dict[key] = value.unsqueeze(0)  # 添加批次维度

    return tensor_dict


if __name__ == "__main__":
    args = parse_args()

    # Load data
    # data = torch.load("./data/square_cloth_test_buf.pt")
    data = torch.load("./chuan/green_test_buf.pt")

    threshold = 5  # 像素阈值
    successful_picks = 0  # 成功的抓取次数
    successful_places = 0  # 成功的放置次数
    total_predictions = len(data)

    use_depth = True
    image_channels = 1 if use_depth else 3
    model = PickToPlaceModel(image_channels, 64).cuda()
    # Load policy
    model.load_state_dict(torch.load(args.weights))
    model.eval()  # 设置为评估模式

    print('len(data)',len(data))
    for sample in data:
        depth_img = sample['obs']['depth']  # 提取深度图
        true_pick = sample['pick']  # 真实的抓取点
        true_place = sample['place']  # 真实的放置点

        batch = process_single_transition(sample, args)
        # batch_tensors = transition_to_tensors(batch)
        pred_pick, pred_place, pick_map, place_map = model.get_pick_place_new(batch)

        # 计算抓取点的欧式距离
        pick_distance = np.sqrt((pred_pick[0] - true_pick[0]) ** 2 + (pred_pick[1] - true_pick[1]) ** 2)
        # 计算放置点的欧式距离
        place_distance = np.sqrt((pred_place[0] - true_place[0]) ** 2 + (pred_place[1] - true_place[1]) ** 2)

        # 判断抓取是否成功
        if pick_distance <= threshold:
            successful_picks += 1
        # 判断放置是否成功
        if place_distance <= threshold:
            successful_places += 1

    pick_accuracy = successful_picks / total_predictions
    place_accuracy = successful_places / total_predictions
    print(f"Pick Accuracy: {pick_accuracy:.2f}")
    print(f"Place Accuracy: {place_accuracy:.2f}")
