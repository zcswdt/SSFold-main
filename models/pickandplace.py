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


# ------------------------------------------------------------------------------
# Network Class
# ------------------------------------------------------------------------------
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


class PickAndPlaceModel(nn.Module):
    def __init__(self, image_channels, image_width):
        super(PickAndPlaceModel, self).__init__()

        self.input_channels = 1
        self.image_width = image_width
        self.device = "cuda"
        self.dim = 256
        self.num_nodes = 200
        graph_encoder_path = 'models/vsbl_edge_best.pth'
        # self.net = FCN(input_channels, 1, bilinear=True)
        self.net = UNetWithTwoDecoders(self.input_channels, 1, 1, 512, bilinear=True)

        # self.resize_graph = FeatureReshapeNet()

        self.resize_graph = ModifyGraphOutput()
        self.se_block = SEBlock(channel=2).to(self.device)
        # graph encoder
        self.graph_encoder = GNN(node_input_dim=3, edge_input_dim=4, proc_layer=10, global_size=128).to(
            device=self.device)
        self.graph_encoder.load_model(graph_encoder_path, self.device)
        self.graph_project = nn.Linear(128, self.dim)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        batch_size = int(graph["x"].shape[0] / self.num_nodes)  # 原始，2
        global_features = torch.zeros(batch_size, 128, dtype=torch.float32, device=self.device)  # 原始，2*128
        graph["u"] = global_features
        # encode
        x = self.graph_encoder(graph)
        x = torch.reshape(x, (-1, self.num_nodes, 128))
        x = self.graph_project(x)  # [batch_size, num_nodes, dim]
        return x

    # --------------------------------------------------------------------------

    def forward(self, obs, pick_map, graph):
        # obs：[16,1,64,64]  pick_map:16[16,1,64,64]
        # get image dimensions
        batch_size, channels, img_height, img_width = obs.shape

        # encode graph features[2,200,512]
        graph_out = self.encode_graph(graph)  # [batch_size, num_nodes, dim]

        x_graph = self.resize_graph(graph_out)

        # combined_features = torch.cat([obs, pick_map], dim=1)
        combined_features = obs

        # 应用SE Block
        # enhanced_features = self.se_block(combined_features)

        # U-Net encoder and decoder
        out = self.net(combined_features, x_graph)  # (b, 1, 64, 64)

        # [16,2,64,64]
        # x = torch.cat([obs, pick_map], dim=1)
        return out

    # --------------------------------------------------------------------------

    def get_loss(self, batch):
        batch['obs'] = batch['obs'].to(self.device)
        pick_heatmaps = batch['pick'].to(self.device)
        place_heatmaps = batch['place'].to(self.device)
        batch['graph'] = batch['graph'].to(self.device)

        outputs = self(batch['obs'], batch['pick'], batch['graph'])
        # outputs = self(batch['obs'], batch['graph'])

        pred_pick_heatmaps = outputs[:, 0, :, :].unsqueeze(1)  # 使用unsqueeze添加通道维度
        place_pick_heatmaps = outputs[:, 1, :, :].unsqueeze(1)  # 使用unsqueeze添加通道维度

        loss_fn = nn.BCEWithLogitsLoss()
        loss_pick = loss_fn(pred_pick_heatmaps, pick_heatmaps)
        loss_place = loss_fn(place_pick_heatmaps, place_heatmaps)
        loss = loss_pick + loss_place

        # loss = torch.mean(F.binary_cross_entropy_with_logits(place_pred, places, reduction="none"))
        return loss

    # --------------------------------------------------------------------------
    def get_pick_place_new(self, batch):
        obs = batch['obs'].to(self.device)
        pick_heatmap = batch['pick'].to(self.device)
        place_heatmaps = batch['place'].to(self.device)
        graph = batch['graph']

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

        outputs = self(obs, pick_heatmap, graph)

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


    # --------------------------------------------------------------------------
    def repeat_graph(self, graph, num_repeats):
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

