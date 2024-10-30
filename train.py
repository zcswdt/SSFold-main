from utils.utils import *
from utils.model_utils import *
import utils.visualize as viz

from models.picktoplace import PickToPlaceModel
from models.pickandplace import PickAndPlaceModel
from models.twonet import TWOModel

import torch
import collections

import random
import torch.nn as nn
import argparse
from tqdm import tqdm
import jsonlines

from utils.graph import build_graph, get_sampled_pc, calc_distances
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt


def sample_batch(dataset, batch_size, args, true_sample=False, fold_only=False):
  raw_batch = random.sample(dataset, batch_size) #
  # batch = collections.defaultdict(list)
  batch = []

  for transition in raw_batch:   #obs,[64,64,3],depth,[64,64] pick-(32, 42) #place (8, 36)  nobs也是一样
    if fold_only:
      # If place not in obs mask, skip
      mask = get_cloth_mask(transition['obs']['rgb'])
      if mask[transition['place'][::-1]] == 0: continue

    batch_elem = {}
    # Store the original images for viz later
    batch_elem['obs_rgb'] = transition['obs']['rgb']/255.0
    batch_elem['nobs_rgb'] = transition['nobs']['rgb']/255.0

    o_mask = transition['obs']['mask']
    # Handle depth or rgb observations, convert to torch tensors
    if args.use_depth:
      obs = transition['obs']['depth']
      obs = torch.FloatTensor(obs).cuda()
      obs = obs.unsqueeze(2)
    else:
      obs = transition['obs']['rgb']/255.0
      obs = torch.FloatTensor(obs).cuda()

    o_mask_tensor = torch.from_numpy(o_mask).unsqueeze(2).cuda()   # 转换后的形状为 [64, 64, 1]



    # 应用遮罩
    obs_masked = obs * o_mask_tensor

    obs_mask = obs_masked.permute(2, 0, 1)

    # Half the time, the sample is a fake negative pick example to balance the positives
    negative_pick_sample = bool(np.random.randint(0, 2)) if args.architecture == "pick_to_place" and not true_sample else False
    if negative_pick_sample:
      pick = get_random_pick(transition['obs']['rgb'])
      place_map = np.zeros((args.image_width,args.image_width))
    else:
      pick = transition['pick']
      place_map = gaussian_heatmap(transition['place'], args.image_width, args.heatmap_sigma)
    pick_map = gaussian_heatmap(pick, args.image_width, args.heatmap_sigma)
    plt.imsave('gaussian_heatmap.png', pick_map, cmap='viridis')

    pick_map  = torch.FloatTensor(pick_map).cuda() .unsqueeze(0)
    place_map = torch.FloatTensor(place_map).cuda().unsqueeze(0)




    #graph,需要构建64*64的图
    depth = transition['obs']['depth']  #需要确认这个depth是 1.原始拍照的图片640*480 2.经过裁减后的图片300*300 3.或者是缩放后的64*64，应该是64*64的
    #64*64
    intrinsics= np.array([[604.99621582, 0.0, 329.07382202],
              [0.0, 605.07873535, 243.38549805],
              [0.0, 0.0, 1.0]])
    camera_pose ='/home/zcs/work/github/folding-by-hand/calibration/cam2table_pose.txt'
    camera_pose = np.loadtxt(camera_pose)


    #需要将相机内参，相机外参传进去，求解从像素坐标-世界坐标的转换camera_params：相机内参
    sampled_pc = get_sampled_pc(
      depth=depth, voxel_size=0.0125, K=200, intrinsics=intrinsics, camera_pose=camera_pose,mask =o_mask)

    graph_data = build_graph(sampled_pc, 0.025)
    graph_data = Data.from_dict(graph_data)





    # Add to batch
    batch_elem['obs'] = obs_mask
    batch_elem['pick'] = pick_map
    batch_elem['place'] = place_map
    # Original pick and place index
    batch_elem['place_index'] = transition['place']
    batch_elem['pick_index'] = transition['pick']
    batch_elem['graph'] =graph_data
    batch_elem['mask'] = transition['obs']['mask']
    batch.append(batch_elem)

  return batch

#------------------------------------------------------------------------------

def get_random_pick(img):
  mask = get_cloth_mask(img)
  indices = get_indices_from_mask(mask)
  pick = choose_random_index(indices)[::-1]
  return pick

#------------------------------------------------------------------------------

# Get batch loss for training
def get_error(batch, args):
  if args.viz: imgs = []
  errors = []
  for batch_elem in batch:
    pred_pick, pred_place, pick_map, place_map = model.get_pick_place(batch_elem)  #前两是坐标，后面两个是热力图
    #pred_pick, pred_place, pick_map, place_map = model.get_pick_place_new(batch_elem)  # 前两是坐标，后面两个是热力图

    if args.viz:
      viz_img = viz.get_viz_img(batch_elem['pick_index'], batch_elem['place_index'], pred_pick, pred_place, batch_elem['obs_rgb'], batch_elem['nobs_rgb'], pick_map, place_map)
      imgs.append(viz_img)
    pred = np.array([pred_pick, pred_place])/args.image_width
    true = np.array([batch_elem['pick_index'], batch_elem['place_index']])/args.image_width

    error = np.mean((true - pred)**2)
    errors.append(error)
  mean_error = np.mean(errors)

  # Images for visualization
  img = None
  if args.viz:
    img = viz.viz_images(imgs, "viz")
  return mean_error, img

#------------------------------------------------------------------------------

def batch_to_tensors(batch):
  # Convert list of dicts to dict of lists
  batch_dict = {k: [d[k] for d in batch] for k in batch[0]}

  # Convert to torch tensors
  for key in batch_dict:
    if key in ["obs_rgb", "nobs_rgb", "pick_index", "place_index","mask"]:
      continue
    if key == "graph":
      # Manually concatenate graph data
      x_list = []
      edge_index_list = []
      edge_attr_list = []
      batch_list = []
      ptr_list = [0]
      current_ptr = 0

      for i, data in enumerate(batch_dict[key]):
        x_list.append(data.x)
        edge_index_list.append(data.edge_index + current_ptr)
        edge_attr_list.append(data.edge_attr)
        batch_list.append(torch.full((data.x.size(0),), i, dtype=torch.long))
        current_ptr += data.x.size(0)
        ptr_list.append(current_ptr)

      batch_dict[key] = Data(
        x=torch.cat(x_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1),
        edge_attr=torch.cat(edge_attr_list, dim=0),
        batch=torch.cat(batch_list, dim=0),
        ptr=torch.tensor(ptr_list, dtype=torch.long)
      )
    else:
      batch_dict[key] = torch.stack(batch_dict[key])

  return batch_dict


#------------------------------------------------------------------------------

def train(dataset, args):
  # Sample batch and get loss
  batch = sample_batch(dataset, args.batch_size, args)    #16个字典[{obs_rgb（64，64，3）,nobs_rgb,obs（1，64，64）,pick（1，64，64）,place,place_index,pick_index},{}...,{}]
  batch_tensors = batch_to_tensors(batch)  #[obs_rgb{16个64*64*3},{16个},obs{16个16*1*64*64},pick{16个16*1*64*64},7个字典]
  #{'obs_rgb','nobs_rgb','obs','pick','place','place_index','pick_index'} 其中pick和place是高斯图
  loss = model.get_loss(batch_tensors)

  # Perform training step
  opt.zero_grad()
  loss.backward()
  opt.step()

  # Log
  if args.wandb:
    wandb.log({"train/loss": loss.item()})

#------------------------------------------------------------------------------

def run_validation(dataset, args):
  global lowest_error
  # Sample validation batch and get loss
  # Loss is BCE for heatmaps,
  # Error is mean squared error for actual pick and place
  batch = sample_batch(dataset, args.batch_size, args)
  batch_tensors = batch_to_tensors(batch)
  loss = model.get_loss(batch_tensors)

  batch = sample_batch(dataset, args.batch_size, args, true_sample=True, fold_only=False)
  error, img = get_error(batch, args)
  # Visualize
  img = img[:,:,::-1]   # 对图像进行色彩通道反转，从BGR转为RGB格式

  # Log
  if args.wandb:
    img = wandb.Image(img, caption="Test Set Visualization")   # 创建W&B图像对象
    wandb.log({"val/val_loss": loss.item(), "val/val_error": error, "val/viz": img})

  # Use error on validation batch to decide when to save model
  #if error < lowest_error:
  lowest_error = error

  # Save model
  if args.save_model:
    filename = "{}{}".format(args.architecture, train_step)
    torch.save(model.state_dict(), "./weights/{}_{}_{}.pt".format(args.name, wandb.run.id, filename))
    with jsonlines.open("./weights/{}_{}_{}.json".format(args.name, wandb.run.id, filename), mode='w') as writer:
      writer.write(vars(args))
      writer.write({"lowest_error": lowest_error})
      writer.write({"train_step": train_step})
  # Log the lowest error metrics/viz
  if args.wandb:
    wandb.log({"best_model/viz": img})  # 记录最佳模型的可视化图像
    wandb.log({"best_model/loss": loss, "best_model/error": error})

#------------------------------------------------------------------------------

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--name", type=str, required=False, help="Name of task",default="cor")
  parser.add_argument("--train_dataset", type=str, required=False, help="Path to training dataset",default="/home/zcs/work/github/ssfold/data_my/huatu/test_buf.pt")
  parser.add_argument("--val_dataset", type=str, required=False, help="Path to val dataset",default="/home/zcs/work/github/ssfold/chuan/red/hunhe/test_buf.pt")
  parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam Optimizer")
  parser.add_argument("--val_interval", type=int, default=100, help="Run validation every n steps")
  parser.add_argument("--training_steps", type=int, default=int(30000), help="Number of training steps")
  parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
  parser.add_argument("--use_depth", type=bool, default=True, help="Use depth for observations")
  parser.add_argument("--image_width", type=int, default=64, help="Image width")
  parser.add_argument("--architecture", type=str, default="pick_to_place", help="Architecture to use for model. Options: twonet, pick_to_place, pick_then_place, pick_and_place")
  parser.add_argument("--one_+step_model", type=bool, default=False, help="Train inverse model instead of multi-step goal model")
  parser.add_argument("--save_model", type=bool, default=True, help="Save model")
  parser.add_argument("--heatmap_sigma", type=int, default=3, help="Sigma for gaussian heatmaps")
  parser.add_argument("--wandb", type=bool, default=True, help="Use wandb for logging")
  parser.add_argument("--viz", type=bool, default=True, help="Visualize training")
  return parser.parse_args()

#------------------------------------------------------------------------------

if __name__ == "__main__":
  args = parse_args()

  train_dataset = torch.load(args.train_dataset)
  print("Training architecture: {}".format(args.architecture))
  print("for {} steps".format(args.training_steps))
  print("Training dataset size: {}".format(len(train_dataset)))
  val_dataset = torch.load(args.val_dataset)

  if args.wandb:
    import wandb
    wandb.init(project="folding_by_hand_{}".format(args.name), config=vars(args))
    wandb.config.update(args)
    wandb.run.name = "{}_{}_{}".format(args.architecture, wandb.run.id, len(train_dataset))

  # Input is obs, pick_map 
  image_channels = 1 if args.use_depth else 3
  if args.architecture == "pick_to_place":
    model = PickToPlaceModel(image_channels, args.image_width).cuda()
  elif args.architecture == "pick_and_place":
    model = PickAndPlaceModel(image_channels, args.image_width).cuda()
  elif args.architecture == "pick_and_place":
    model = TWOModel(image_channels, args.image_width).cuda()

  opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  lossfn = nn.BCELoss()

  lowest_error = np.inf # Starting point for lowest error
  for train_step in tqdm(range(args.training_steps)):
    train(train_dataset, args)
    if train_step % args.val_interval == 0 and train_step > 0:
      with torch.no_grad():
        run_validation(val_dataset, args)
