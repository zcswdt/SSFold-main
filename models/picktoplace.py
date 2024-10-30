import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import cv2

from utils import utils
from .net import FCN
from .unet import UNet

from .twode import UNetWithTwoDecoders

from .vcd import GNN
from torch_geometric.data import Data, Batch
#------------------------------------------------------------------------------
# Network Class
#------------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 压缩
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 激励
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Squeeze
        y = self.fc(y).view(b, c, 1, 1)  # Excitation
        return x * y.expand_as(x)  # Scale

class FeatureReshapeNet(nn.Module):
    def __init__(self):
        super(FeatureReshapeNet, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))
        self.conv = nn.Conv2d(1, 1, kernel_size=1)  # 1x1卷积用于调整通道数

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度
        x = self.adaptive_pool(x)  # 自适应池化调整空间尺寸
        x = self.conv(x)  # 调整通道数
        return x


class ModifyGraphOutput(nn.Module):
    def __init__(self):
        super(ModifyGraphOutput, self).__init__()
        # 使用1x1卷积来调整通道数从200到512
        self.conv1x1 = nn.Conv2d(200, 512, kernel_size=1)
        # 使用适应性平均池化来改变空间尺寸到4x4
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        # 假设输入x的形状是[batch_size, channels, height]
        # 先添加一个假的宽度维度
        x = x.unsqueeze(-1)  # 变为[batch_size, channels, height, 1]

        # 调整通道数
        x = self.conv1x1(x)

        # 调整空间尺寸
        x = self.adaptive_pool(x)
        return x


class PickToPlaceModel(nn.Module):
    def __init__(self, image_channels, image_width):
        super(PickToPlaceModel, self).__init__()

        #input_channels = image_channels + 2   #因为有三个特征
        self.image_width = image_width
        self.device = "cuda"
        self.dim = 256
        self.num_nodes = 200
        graph_encoder_path='models/vsbl_edge_best.pth'
        # self.net = FCN(input_channels, 1, bilinear=True)
        self.net = UNetWithTwoDecoders(2, 1,1,512, bilinear=True)

        #self.resize_graph = FeatureReshapeNet()

        self.resize_graph = ModifyGraphOutput()
        self.se_block = SEBlock(channel=2).to(self.device)
        # graph encoder
        self.graph_encoder = GNN(node_input_dim=3, edge_input_dim=4, proc_layer=10, global_size=128).to(
            device=self.device)
        self.graph_encoder.load_model(graph_encoder_path, self.device)
        self.graph_project = nn.Linear(128, self.dim)
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # freeze graph_encoder
        self.frozen_graph_encoder()

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path)["model"])
        print(f"load agent from {model_path}")

    def frozen_graph_encoder(self):
        for param in self.named_parameters():
            if "graph_encoder" in param[0]:
                param[1].requires_grad = False

    def frozen_all(self):
        for param in self.named_parameters():
            param[1].requires_grad = False

    def encode_graph(self, graph):
        # add initial global features
        batch_size = int(graph["x"].shape[0] / self.num_nodes)  #原始，2
        global_features = torch.zeros(batch_size, 128, dtype=torch.float32, device=self.device)  #原始，2*128
        graph["u"] = global_features
        # encode
        x = self.graph_encoder(graph)
        x = torch.reshape(x, (-1, self.num_nodes, 128))
        x = self.graph_project(x)  # [batch_size, num_nodes, dim]
        return x

    #--------------------------------------------------------------------------

    def forward(self, obs, pick_map, graph):
        #obs：[16,1,64,64]  pick_map:16[16,1,64,64]
        # get image dimensions
        batch_size, channels, img_height, img_width = obs.shape

        #encode graph features[2,200,512]
        graph_out = self.encode_graph(graph)  # [batch_size, num_nodes, dim]

        x_graph = self.resize_graph(graph_out)


        combined_features = torch.cat([obs, pick_map], dim=1)

        # 应用SE Block
        #enhanced_features = self.se_block(combined_features)

        # U-Net encoder and decoder
        out = self.net(combined_features,x_graph)  #(b, 1, 64, 64)

        #[16,2,64,64]
        #x = torch.cat([obs, pick_map], dim=1)
        return out


    #--------------------------------------------------------------------------

    def get_loss(self, batch):
        batch['obs'] = batch['obs'].to(self.device)
        pick_heatmaps = batch['pick'].to(self.device)
        place_heatmaps = batch['place'].to(self.device)
        batch['graph'] = batch['graph'].to(self.device)

        outputs  = self(batch['obs'], batch['pick'],batch['graph'])
        #outputs = self(batch['obs'], batch['graph'])

        pred_pick_heatmaps = outputs[:, 0, :, :].unsqueeze(1)  # 使用unsqueeze添加通道维度
        place_pick_heatmaps = outputs[:, 1, :, :].unsqueeze(1)  # 使用unsqueeze添加通道维度


        # #是pick and place的
        # loss_fn = nn.BCEWithLogitsLoss()
        # loss_pick = loss_fn(pred_pick_heatmaps, pick_heatmaps)
        # loss_place = loss_fn(place_pick_heatmaps, place_heatmaps)
        # loss = loss_pick + loss_place


        loss = torch.mean(F.binary_cross_entropy_with_logits(pred_pick_heatmaps, place_heatmaps, reduction="none"))
        return loss


    #--------------------------------------------------------------------------
    def get_pick_place_new(self, batch):
        obs= batch['obs'].to(self.device)
        pick_heatmap = batch['pick'].to(self.device)
        place_heatmaps = batch['place'].to(self.device)
        graph= batch['graph']

        graph = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=torch.zeros(graph.x.size(0), dtype=torch.long),
            ptr=torch.tensor([0, graph.x.size(0)], dtype=torch.long)
        )

        graph = graph.to(self.device)

        obs = obs.unsqueeze(1)  # 1 表示新维度插入的位置
        pick_heatmap = pick_heatmap.unsqueeze(1)

        outputs  = self(obs, pick_heatmap, graph)

        heatmaps = F.sigmoid(outputs)

        pred_pick_heatmaps = heatmaps[:, 0, :, :].unsqueeze(1)  # 使用unsqueeze添加通道维度(1,1,h,w)
        place_pick_heatmaps = heatmaps[:, 1, :, :].unsqueeze(1)  # 使用unsqueeze添加通道维度(1,1,h,w)

        # 去除所有单一维度
        pred_pick_heatmaps = pred_pick_heatmaps.squeeze().detach().cpu().numpy()
        place_pick_heatmaps = place_pick_heatmaps.squeeze().detach().cpu().numpy()


        pred_pick = np.array(np.unravel_index(np.argmax(pred_pick_heatmaps), pred_pick_heatmaps.shape))
        pred_pick = pred_pick[::-1]
        pred_place = np.array(np.unravel_index(np.argmax(place_pick_heatmaps), place_pick_heatmaps.shape))
        pred_place = pred_place[::-1]

        return pred_pick, pred_place, pred_pick_heatmaps, place_pick_heatmaps



    def get_pick_place(self, batch_elem):
        obs = batch_elem['obs']  #只有布料区域得图像
        mask = batch_elem['mask']
        graph = batch_elem['graph']


        obs_rgb = obs.permute(1,2,0).detach().cpu().numpy() # Kinda hacky...
        pick_gauss, pick_indices = get_pickmaps_from_obs(obs_rgb, self.image_width)  #在布料区域下采样找到所有可能的pick点，然后生成高斯热力图，以及相应的坐标



        place_maps = []
        i = 0
        batch_size = 100
        while i < len(pick_gauss):
            batch = pick_gauss[i:i+batch_size]
            torch.cuda.empty_cache()
            place_map = torch.sigmoid(self._get_place_maps(obs, batch, graph)).squeeze(0).detach().cpu().numpy()   #将观察图和相应的可能 抓取点输入到网络，得到相应的放置点
            #print("place_map", place_map.shape)
            if len(batch) == 1:
                place_map = np.expand_dims(place_map, axis=0)
            place_maps.append(place_map)

            # print(len(place_maps))
            i += batch_size
        place_maps = np.concatenate(place_maps, axis=0)


        # place_maps = []
        # for pick in pick_gauss:
        #     print(i)
        #     pick = np.expand_dims(pick, axis=0)
        #     place_map = torch.sigmoid(self._get_place_maps(obs, pick)).squeeze(0).detach().cpu().numpy()
        #     place_maps.append(place_map)
        #     i += 1
        # place_maps = np.stack(place_maps)
        # print("place_maps", place_maps.shape)
        # for i in range(len(place_maps)):
        #     print(np.max(place_maps[i]))
        #     print(np.min(fast_place_maps[i]))
        #     cv2.imshow("place_map", place_maps[i][0])
        #     cv2.imshow("fast_place_map", fast_place_maps[i][0])
        #     cv2.waitKey(0)
        #在pick_map上所有可能的抓取点上，得到place的点，并且得到最优的pick点的索引
        pick_map, best_pick_num = self._get_pick_map(place_maps, pick_indices)#从网络预测的所有放置点中，反推断出最好的抓取热力图,pick(64,64)
        pred_pick, pred_place = self._get_pick_place_inds_from_maps(pick_map, place_maps, best_pick_num)  #预测的抓取点和放置点的坐标
        best_place_map = place_maps[best_pick_num].squeeze(0)#.detach().cpu().numpy()  #最好的放置点的热力图
        return pred_pick, pred_place, pick_map, best_place_map   #

    def _get_pick_place_inds_from_maps(self, pick_map, place_maps, best_pick_num):
        # Unravel pick location
        pick = np.array(np.unravel_index(np.argmax(pick_map), pick_map.shape))
        pick = pick[::-1]

        # Unravel place location
        # place_map = place_maps[best_pick_num].squeeze(0).detach().cpu().numpy()
        place_map = place_maps[best_pick_num].squeeze(0)
        place = np.array(np.unravel_index(np.argmax(place_map), place_map.shape))
        place = place[::-1]
        return pick, place
    #--------------------------------------------------------------------------
    def repeat_graph(self,graph, num_repeats):
        """
        Repeat a graph `num_repeats` times and return the repeated graph.

        Parameters:
        - graph: A PyTorch Geometric `Data` object representing the graph.
        - num_repeats: The number of times to repeat the graph.
        - device: The device to which the graph should be moved (e.g., 'cuda' or 'cpu').

        Returns:
        - A PyTorch Geometric `Data` object representing the repeated graph.
        """
        x_list = [graph.x for _ in range(num_repeats)]
        edge_index_list = [graph.edge_index for _ in range(num_repeats)]
        edge_attr_list = [graph.edge_attr for _ in range(num_repeats)]

        repeated_x = torch.cat(x_list, dim=0)
        repeated_edge_index = torch.cat([ei + i * graph.x.size(0) for i, ei in enumerate(edge_index_list)], dim=1)
        repeated_edge_attr = torch.cat(edge_attr_list, dim=0)
        repeated_batch = torch.cat([torch.full((graph.x.size(0),), i, dtype=torch.long) for i in range(num_repeats)],
                                   dim=0)

        repeated_graph = Data(
            x=repeated_x,
            edge_index=repeated_edge_index,
            edge_attr=repeated_edge_attr,
            batch=repeated_batch
        )

        return repeated_graph


    def _get_place_maps(self, obs, pick_gauss,graph):
        obs = obs.repeat(len(pick_gauss),1,1,1)#.permute(0,3,1,2)  重复生成同等数量的pick_gauss的数量
        picks = torch.FloatTensor(np.array(pick_gauss)).cuda().unsqueeze(1)
        repeated_graph = self.repeat_graph(graph, len(pick_gauss))
        obs = obs.to(self.device)
        picks = picks.to(self.device)
        repeated_graph = repeated_graph.to(self.device)


        place_maps = self(obs, picks, repeated_graph)  #将其输入网络，输出很多个抓取点
        place_pick_heatmaps = place_maps[:, 0, :, :]
        place_map = place_pick_heatmaps.unsqueeze(1)
        return place_map
    
    #--------------------------------------------------------------------------

    def _get_pick_map(self, place_maps, pick_indices):
      pickmap = np.zeros((self.image_width,self.image_width))   #初始化一个图像宽度大小的零矩阵用作拾取图
      maxes = []  # 用于存储每个位置最大值的列表
      for ind, place_maps in zip(pick_indices, place_maps):
        place_maps = place_maps#.detach().cpu().numpy()
        q_max = np.max(place_maps)  # 找出每个放置图的最大值
        maxes.append(q_max)         # 将最大值添加到列表中
        pickmap[ind[0],ind[1]] = q_max  # 在拾取图中相应的索引位置设置这个最大值
        # # draw circle at pick loc ation
        # circle_im = np.zeros((self.image_width,self.image_width))
        # cv2.circle(circle_im, tuple(ind[::-1]), 5, (1.0,1.0,1.0), -1)
        # circle_im = circle_im*q_max
        # pickmap = pickmap + circle_im
        # if debug:
        #   cv2.imshow("circle", circle_im)
        #   cv2.imshow("pickmap", pickmap)
        #   cv2.waitKey(0)

      best_pick_num = np.argmax(maxes)    #找到最大值列表中的最大值的索引，即最佳拾取点的编号
      return pickmap, best_pick_num    #返回拾取图和最佳拾取点的编号


#------------------------------------------------------------------------------

def get_pickmaps_from_obs(obs, image_width, heatmap_sigma=2):
  mask = utils.get_cloth_mask(obs)

  # indices = get_indices_from_mask(mask>0) 按照给定的图像宽度进行下采样，可能是为了减少处理数据量或适应处理分辨率。
  indices = get_downsampled_indices_from_mask(mask, image_width)
  possible_picks = indices

  pick_maps = []  #在布料mask区域找到所有可能pick点，然后将这些可能的pick点，生成一个高斯热力图
  for index in possible_picks:
    pick_map = utils.gaussian_heatmap(index[::-1], image_width, heatmap_sigma)  # index[::-1]调整索引顺序（通常从 (x, y) 转换为 (y, x)）
    pick_maps.append(pick_map)
    # debug=True
    # if debug:
    #   cv2.imshow("pick", pick_map)
    #   cv2.waitKey(0)

  return pick_maps, indices #, possible_dwnsc

#------------------------------------------------------------------------------
#这个函数通过下采样和筛选有效地减少了处理的点数量，提高了后续处理步骤（如生成高斯热图等）的效率。
def get_downsampled_indices_from_mask(mask, image_width, downsample_scale=2):
  # Create a grid of points evenly spaced across the image, 20x20
  grid = np.mgrid[0:image_width:downsample_scale,
                  0:image_width:downsample_scale].reshape(2,-1).T
  # Reverse x and y to match the mask
  grid = grid[:, [1,0]]
  # Remove the grid points that are not in the mask
  grid = grid[mask[grid[:,0], grid[:,1]] == 1]
  return grid

