# Copyright (c) Facebook, Inc. and its affiliates.
from torch import nn
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone import FPN
import math
import torch.nn.functional as F
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.data import build_detection_train_loader
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.samplers.distributed_sampler import TrainingSampler
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.transforms as T

import time
import random
import torch
import PIL.Image as Image
import os



class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        # 定义残差块中的卷积层

        self.conv1 = nn.Conv2d(num_channels ,num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels ,num_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # 保存残差连接的输入
        identity = x
        out_1 = self.conv1(x)
        # 第一个卷积层
        out_1 = F.relu(out_1)
        # 第二个卷积层
        out_2 = self.conv2(out_1)
        # 残差连接
        out_2 += identity
        out = F.relu(out_2, inplace=True)
        return out


class Upsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(  # nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(n_feat*2),
            # nn.Hardswish(),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
#全局平均池化
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAttentionModule, self).__init__()
        # 定义一个1x1的卷积层，可以改变特征的通道数
        self.conv1x1 = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        # 平均池化
        avg_pool = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # 最大池化
        max_pool = nn.functional.adaptive_max_pool2d(x, (1, 1))
        # 拼接平均池化和最大池化的结果
        pool_concat = torch.cat([avg_pool, max_pool], dim=1)
        # 通过1x1卷积层
        conv1x1 = self.conv1x1(pool_concat)
        sa_map = torch.sigmoid(conv1x1)
        return sa_map


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        maps = self.sigmoid(x)

        return maps



def build_train_aug(short_edge_length,max_size,sample_style,fixed_seed):
    augs = [
        T.ResizeShortestEdge(short_edge_length,max_size,sample_style,fixed_seed=fixed_seed)]
    return augs
def save_images(tensor, num_channels, img_dir):
    """
    将一个多通道张量分解成单独的影像，并保存到指定的文件夹中。

    参数:
    tensor (torch.Tensor): 输入的多通道张量，形状应为 [1, num_channels, height, width]。
    num_channels (int): 通道的数量，即要保存的影像数量。
    img_dir (str): 保存影像的文件夹路径。
    """
    # 确保文件夹存在，如果不存在则创建
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # 遍历通道并保存每个通道的图像
    for i in range(num_channels):
        # 提取当前通道的影像，确保不保留梯度，并移动到CPU
        channel_image = tensor[0, i, :, :].detach().cpu().numpy()

        # 如果通道数据是浮点数，需要将其缩放到0到255的范围内
        if channel_image.max() > 1:
            channel_image = (channel_image - channel_image.min()) / (channel_image.max() - channel_image.min()) * 255

        # 将图像数据转换为PIL图像
        img = Image.fromarray(channel_image.astype('uint8'), 'L')  # 使用灰度模式 'L'
        random_str = str(random.randint(1, 1000))
        timestamp = int(time.time())
        # 保存影像
        filename = os.path.join(img_dir, f'img_{timestamp}_{random_str}_{i}.png')
        img.save(filename)



def tensor_to_image(tensor, mode='RGB'):
    """
    将0-255范围内的张量转换为PIL图像。

    参数:
    - tensor (Tensor): 输入的图像张量，形状应为 [H, W] 或 [C, H, W]，其中 C 是通道数。
    - mode (str): PIL图像模式，例如 'RGB'、'L'（灰度）等。默认为 'RGB'。

    返回:
    - PIL.Image: 转换后的图像。
    """
    # 确保张量在 [0, 255] 范围内
    tensor = torch.clamp(tensor, 0, 255).byte()

    # 根据张量的维度处理
    if tensor.dim() == 2:  # 灰度图或单通道图像
        image = Image.fromarray(tensor.cpu().numpy(), mode=mode)
    elif tensor.dim() == 3:  # 彩色图像
        # 确保通道顺序正确
        if len(tensor.size()) == 3 and tensor.size(0) == 3:
            tensor = tensor.permute(1, 2, 0)  # 从 [C, H, W] 转换为 [H, W, C]
        image = Image.fromarray(tensor[0].cpu().numpy(), mode=mode) if tensor.size(0) == 1 else Image.fromarray(tensor.cpu().numpy(), mode=mode)
    else:
        raise ValueError("Unsupported tensor dimension for image conversion.")

    return image
def denormalize_image(image_normalized, pixel_mean, pixel_std):
    """
    Denormalize the image by reversing the normalization process.

    :param image_normalized: The normalized image tensor with values in the range [0, 1].
    :param pixel_mean: The mean values used for normalization for each channel.
    :param pixel_std: The standard deviation values used for normalization for each channel.
    :return: The denormalized image tensor with pixel values in the range [0, 255].
    """

    # Convert pixel_mean and pixel_std to tensors with the same device as image_normalized
    mean = torch.tensor(pixel_mean).to(image_normalized.device).view(3, 1, 1)
    std = torch.tensor(pixel_std).to(image_normalized.device).view(3, 1, 1)
    # image_scaled = (image_normalized * 255).to(torch.uint8)
    # Reverse the normalization process
    image_denormalized = (image_normalized * std + mean)

    # Scale the values to the [0, 255] range
     # Convert to 8-bit unsigned integer

    return image_denormalized


# Example usage:
  # Example normalized image tensor
pixel_mean = [123.675, 116.280, 103.530]
pixel_std = [58.395, 57.120, 57.375]
# pixel_mean = [0, 0, 0]
# pixel_std = [255, 255, 255]
# class ImagePairDataset(Dataset):
#     def __init__(self, img_dir1, img_dir2, transform=None):
#         self.img_dir1 = img_dir1
#         self.img_dir2 = img_dir2
#         self.transform = transform
#         self.images1 = os.listdir(img_dir1)
#         self.images2 = os.listdir(img_dir2)
#
#         # 确保两个文件夹中的图像数量相同
#         assert len(self.images1) == len(self.images2), "图像数量不匹配"
#
#     def __len__(self):
#         return len(self.images1)
#
#     def __getitem__(self, idx):
#         img_path1 = os.path.join(self.img_dir1, self.images1[idx])
#         img_path2 = os.path.join(self.img_dir2, self.images2[idx])
#
#         image1 = Image.open(img_path1).convert('RGB')
#         image2 = Image.open(img_path2).convert('RGB')
#
#         if self.transform:
#             image1 = self.transform(image1)
#             image2 = self.transform(image2)
#
#         return image1, image2




def save_image_from_tensor(tensor, save_path, normalize=True):
    """
    将PyTorch张量转换为RGB图像并保存到指定路径。

    参数:
    - tensor (Tensor): 输入的PyTorch张量，应为[batch_size, C, H, W]格式，其中C=3。
    - save_path (str): 保存图像的路径。
    - normalize (bool): 是否对张量进行归一化处理，将其范围从[0, 1]转换为[0, 255]，默认为True。
    """
    # 确保张量在CPU上
    tensor = tensor.detach().cpu()

    # 检查张量是否为RGB格式（C=3）
    # if tensor.size(1) != 3:
    #     raise ValueError("The tensor must have 3 color channels (RGB).")

    # 如果需要，对张量进行归一化处理
    if normalize:
        tensor = tensor.clamp(-2.1, -2.05)  # 将值限制在[0, 1]之间
        # 转换为[0, 255]并转换为uint8类型
        tensor = ((tensor * 255).to(torch.uint8)).squeeze()

    # 将张量转换为PIL图像
    # 注意：确保转换后的张量维度顺序为HxWxC
    img_pil = Image.fromarray(tensor.numpy().transpose(1, 2, 0))

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像到指定路径
    img_pil.save(save_path)
#特征融合模块

#去雾网络
class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        self.b = 1

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3),1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4),1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        # save_image_from_tensor(F.relu(output), f'E:\\output-hazy-middle\\image.jpg' , normalize=True)
        return F.relu(output)

# BACKBONE_REGISTRY = Registry("BACKBONE")

class MyCustomBackbone1(Backbone):
    def __init__(
            self,
            bottom_up,
            in_features,
            out_channels,
            norm="",
            top_block=None,
            fuse_type="sum",
            square_pad=0,
    ):
        super(MyCustomBackbone1, self).__init__()
        self.FPN = FPN(bottom_up,
                      in_features ,
                      out_channels ,
                      norm,
                      top_block,
                      fuse_type,
                      )

        assert isinstance(bottom_up, Backbone)
        assert in_features, in_features
        input_shapes = bottom_up.output_shape()
        strides = [input_shapes[f].stride for f in in_features]
        in_channels_per_feature = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        # use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

        self.in_features = tuple(in_features)
        # self.bottom_up1 = bottom_up1
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        # if self.top_block is not None:
        #     for s in range(stage, stage + self.top_block.num_levels):
        #         self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self.AOD = AODnet()

    def forward(self, x):

        dehazing_result = self.AOD(x)
        # output = self.FPN(x)
        # output,dehazing_result = self.FPN(x)
        output = self.FPN(dehazing_result)

        return output, dehazing_result



class MyCustomBackbone(Backbone):
    def __init__(
            self,
            cfg,
            input_shape
    ):
        super(MyCustomBackbone, self).__init__()
        #head输入
        self._out_feature_strides = {'res5': 64}
        self._out_features = list(['res5'])
        self._out_feature_channels = {'res5': 2048}
        #共享encoder
        self.Resnet = build_resnet_backbone(cfg,input_shape)

    def forward(self, x):
        Resnet = self.Resnet(x)
        stem = Resnet['stem']
        res2 = Resnet['res2']
        res3 = Resnet['res3']
        res4 = Resnet['res4']
        res5 = Resnet['res5']


        return Resnet

@BACKBONE_REGISTRY.register()
def build_dehazing_resnet_backbone(cfg, input_shape: ShapeSpec):

    backbone = MyCustomBackbone(cfg,input_shape)
    return backbone


@BACKBONE_REGISTRY.register()
def build_dehazing_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_resnet_backbone(cfg, input_shape)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = MyCustomBackbone1(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,)

    return backbone

def get_dehazing_loss(hazy_image, clear_image):
    # 创建损失函数实例
    criterion = nn.L1Loss()
    # 计算损失
    loss_hazy = criterion(hazy_image, clear_image)
    return loss_hazy

def build_clear_dataset(datasets_name_clear):

    dataset_clear = get_detection_dataset_dicts(datasets_name_clear)
    sampler = TrainingSampler(len(dataset_clear), shuffle=True, seed=40244023)
    mapper = DatasetMapper(is_train=True, image_format="RGB", augmentations=build_train_aug((512),1333,
                                                                                                    "choice",42))
    dataset_clear_loader = build_detection_train_loader(
        dataset=dataset_clear,
        mapper=mapper,
        sampler=sampler,
        total_batch_size=1,
        aspect_ratio_grouping=False,
        num_workers=1,
        collate_fn=None
    )
    return dataset_clear_loader

def get_next_clear_image(clear_dataset_loader):
    for batch_clear in clear_dataset_loader:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        yield batch_clear[0]['image'].to(device)



