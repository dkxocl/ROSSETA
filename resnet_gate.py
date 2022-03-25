import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
import copy

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.backbone import Backbone
import math
import torch.distributed as dist
__all__ = [
    "ResNetBlockBase",
    "BasicBlock_Gate",
    "BottleneckBlock_Gate",
    "DeformBottleneckBlock",
    "BasicStem_Gate",
    "ResNet_Gate",
    "make_stage",
    "build_resnet_backbone_gate",
]


class BasicBlock_Gate(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, in_channels, out_channels, *, stride=1, norm="BN"):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BottleneckBlock_Gate(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            bottleneck_channels,
            gate=False,
            pre_class=0,
            num_class=80,
            class_embedding=None,
            stride=1,
            num_groups=1,
            norm="BN",
            stride_in_1x1=False,
            dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)
        self.gate = gate
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bottleneck_channels = bottleneck_channels
        if gate != False:
            self.class_embedding_total = class_embedding.cuda()
            self.class_embedding = self.class_embedding_total[pre_class:pre_class + num_class][:]

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
            if gate:
                self.gate_short_fc1 = nn.Linear(in_channels, 16)
                self.gate_short_fc2 = nn.Linear(16, out_channels)
                self.gate_short_class_embedding_fc = nn.Linear(300, in_channels)
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )
        if gate:
            self.gate1_fc1 = nn.Linear(in_channels, 16)
            self.gate1_fc2 = nn.Linear(16, bottleneck_channels)
            self.gate1_class_embedding_fc = nn.Linear(300, in_channels)

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        if gate:
            self.gate2_fc1 = nn.Linear(bottleneck_channels, 16)
            self.gate2_fc2 = nn.Linear(16, bottleneck_channels)
            self.gate2_class_embedding_fc = nn.Linear(300, bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),

        )
        if gate:
            self.gate3_fc1 = nn.Linear(bottleneck_channels, 16)
            self.gate3_fc2 = nn.Linear(16, out_channels)
            self.gate3_class_embedding_fc = nn.Linear(300, bottleneck_channels)

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x, fre_gate_list=None, fre_gate=None,pre_gate_list=None ):
        gate_loss = 0
        gate_list = []
        diversity_loss_total = []
        number = 0
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
            if self.gate:
                if fre_gate == None:
                    gate_short = F.adaptive_avg_pool2d(x, (1, 1))
                    gate_short = gate_short.view(-1, self.in_channels)
                    class_embedding = self.gate_short_class_embedding_fc(self.class_embedding)
                    class_embedding = torch.max(class_embedding, 0)[0]
                    class_embedding = class_embedding.unsqueeze(0)
                    class_embedding = class_embedding.repeat(gate_short.shape[0], 1)
                    gate_short = F.relu(gate_short + class_embedding)
                    gate_short = self.gate_short_fc1(gate_short)
                    gate_short = F.relu(gate_short)
                    gate_short = self.gate_short_fc2(gate_short)
                    noise_short = (-torch.empty_like(gate_short).exponential_().log())
                    gate_short = noise_short + gate_short
                    gate_short = torch.sigmoid(gate_short)
                    gate_loss = gate_loss + gate_short.sum() / self.out_channels
                    gate_list.append(gate_short.sum(dim=0))
                    shortcut = shortcut.permute(2, 3, 0, 1)
                    shortcut = shortcut * gate_short
                    shortcut = shortcut.permute(2, 3, 0, 1)
                    if pre_gate_list != None:
                        for i in range(len(pre_gate_list)):
                            pre_gate = pre_gate_list[i].pop(0).cuda()
                            pre_gate = pre_gate.unsqueeze(0).repeat(gate_short.shape[0], 1)
                            a = torch.ones_like(pre_gate)
                            b = torch.zeros_like(pre_gate)
                            con_pre_gate = torch.where(pre_gate == 1, b, a)
                            mid_gate = torch.where(con_pre_gate == 0, b, gate_short)
                            now_gate = torch.where(mid_gate > 0.5, a, b)
                            nnow_gate = now_gate - mid_gate.detach() + mid_gate
                            p = (nnow_gate.sum(dim=1) + 0.01) / (con_pre_gate.sum(dim=1) + 0.1)
                            diversity_loss = (p.mul(p.log()) + (1 - p).mul((1 - p).log()))
                            if number == 0 :
                                diversity_loss_total.append(diversity_loss)
                            else:
                                diversity_loss_total[i] += diversity_loss
                        number += 1



                else:
                    gate_short = fre_gate_list.pop(0).cuda()
                    gate_short = gate_short.unsqueeze(0)
                    gate_short = gate_short.unsqueeze(-1)
                    gate_short = gate_short.unsqueeze(-1)
                    shortcut = gate_short * shortcut

        else:
            shortcut = x

        out = self.conv1(x)
        if self.gate:
            if fre_gate == None:
                gate1 = F.adaptive_avg_pool2d(x, (1, 1))
                gate1 = gate1.view(-1, self.in_channels)
                class_embedding = self.gate1_class_embedding_fc(self.class_embedding)
                class_embedding = torch.max(class_embedding, 0)[0]
                class_embedding = class_embedding.unsqueeze(0)
                class_embedding = class_embedding.repeat(gate1.shape[0], 1)
                gate1 = F.relu(gate1 + class_embedding)
                gate1 = self.gate1_fc1(gate1)
                gate1 = F.relu(gate1)
                gate1 = self.gate1_fc2(gate1)
                noise1 = (-torch.empty_like(gate1).exponential_().log())
                gate1 = noise1 + gate1
                gate1 = torch.sigmoid(gate1)
                gate_loss = gate_loss + gate1.sum() / self.bottleneck_channels
                gate_list.append(gate1.sum(dim=0))
                out = out.permute(2, 3, 0, 1)
                out = out * gate1
                out = out.permute(2, 3, 0, 1)
                if pre_gate_list != None:
                    for i in range(len(pre_gate_list)):
                        pre_gate = pre_gate_list[i].pop(0).cuda()
                        pre_gate = pre_gate.unsqueeze(0).repeat(gate1.shape[0], 1)
                        a = torch.ones_like(pre_gate)
                        b = torch.zeros_like(pre_gate)
                        con_pre_gate = torch.where(pre_gate == 1, b, a)
                        mid_gate = torch.where(con_pre_gate == 0, b, gate1)
                        now_gate = torch.where(mid_gate > 0.5, a, b)
                        nnow_gate = now_gate - mid_gate.detach() + mid_gate
                        p = (nnow_gate.sum(dim=1) + 0.01) / (con_pre_gate.sum(dim=1) + 0.1)
                        diversity_loss = (p.mul(p.log()) + (1 - p).mul((1 - p).log()))
                        if number == 0:
                            diversity_loss_total.append(diversity_loss)
                        else:
                            diversity_loss_total[i] += diversity_loss
                    number += 1
            else:
                gate1 = fre_gate_list.pop(0).cuda()
                gate1 = gate1.unsqueeze(0)
                gate1 = gate1.unsqueeze(-1)
                gate1 = gate1.unsqueeze(-1)
                out = gate1 * out
        out = F.relu_(out)

        out2 = self.conv2(out)
        if self.gate:
            if fre_gate == None:
                gate2 = F.adaptive_avg_pool2d(out, (1, 1))
                gate2 = gate2.view(-1, self.bottleneck_channels)
                class_embedding = self.gate2_class_embedding_fc(self.class_embedding)
                class_embedding = torch.max(class_embedding, 0)[0]
                class_embedding = class_embedding.unsqueeze(0)
                class_embedding = class_embedding.repeat(gate2.shape[0], 1)
                gate2 = F.relu(gate2 + class_embedding)
                gate2 = self.gate2_fc1(gate2)
                gate2 = F.relu(gate2)
                gate2 = self.gate2_fc2(gate2)
                noise2 = (-torch.empty_like(gate2).exponential_().log())
                gate2 = noise2 + gate2
                gate2 = torch.sigmoid(gate2)
                gate_loss = gate_loss + gate2.sum() / self.bottleneck_channels
                gate_list.append(gate2.sum(dim=0))
                out2 = out2.permute(2, 3, 0, 1)
                out2 = out2 * gate2
                out2 = out2.permute(2, 3, 0, 1)
                if pre_gate_list != None:
                    for i in range(len(pre_gate_list)):
                        pre_gate = pre_gate_list[i].pop(0).cuda()
                        pre_gate = pre_gate.unsqueeze(0).repeat(gate2.shape[0], 1)
                        a = torch.ones_like(pre_gate)
                        b = torch.zeros_like(pre_gate)
                        con_pre_gate = torch.where(pre_gate == 1, b, a)
                        mid_gate = torch.where(con_pre_gate == 0, b, gate2)
                        now_gate = torch.where(mid_gate > 0.5, a, b)
                        # print(f'now_gate.sum(){now_gate.sum()}')
                        # print(f'con_pre_gate.sum(){con_pre_gate.sum()}')
                        nnow_gate = now_gate - mid_gate.detach() + mid_gate
                        p = (nnow_gate.sum(dim=1) + 0.01) / (con_pre_gate.sum(dim=1) + 0.1)
                        # print(f'p{p}')
                        diversity_loss = (p.mul(p.log()) + (1 - p).mul((1 - p).log()))
                        if number == 0:
                            diversity_loss_total.append(diversity_loss)
                        else:
                            diversity_loss_total[i] += diversity_loss
                    number += 1
            else:
                gate2 = fre_gate_list.pop(0).cuda()
                gate2 = gate2.unsqueeze(0)
                gate2 = gate2.unsqueeze(-1)
                gate2 = gate2.unsqueeze(-1)
                out2 = gate2 * out2

        out2 = F.relu_(out2)

        out3 = self.conv3(out2)
        if self.gate:
            if fre_gate == None:
                gate3 = F.adaptive_avg_pool2d(out2, (1, 1))
                gate3 = gate3.view(-1, self.bottleneck_channels)
                class_embedding = self.gate3_class_embedding_fc(self.class_embedding)
                class_embedding = torch.max(class_embedding, 0)[0]
                class_embedding = class_embedding.unsqueeze(0)
                class_embedding = class_embedding.repeat(gate3.shape[0], 1)
                gate3 = F.relu(gate3 + class_embedding)
                gate3 = self.gate3_fc1(gate3)
                gate3 = F.relu(gate3)
                gate3 = self.gate3_fc2(gate3)
                noise3 = (-torch.empty_like(gate3).exponential_().log())
                gate3 = noise3 + gate3
                gate3 = torch.sigmoid(gate3)
                gate_loss = gate_loss + gate3.sum() / self.out_channels
                gate_list.append(gate3.sum(dim=0))
                out3 = out3.permute(2, 3, 0, 1)
                out3 = out3 * gate3
                out3 = out3.permute(2, 3, 0, 1)
                if pre_gate_list != None:
                    for i in range(len(pre_gate_list)):
                        pre_gate = pre_gate_list[i].pop(0).cuda()
                        pre_gate = pre_gate.unsqueeze(0).repeat(gate3.shape[0], 1)
                        a = torch.ones_like(pre_gate)
                        b = torch.zeros_like(pre_gate)
                        con_pre_gate = torch.where(pre_gate == 1, b, a)
                        mid_gate = torch.where(con_pre_gate == 0, b, gate3)
                        now_gate = torch.where(mid_gate > 0.5, a, b)
                        nnow_gate = now_gate - mid_gate.detach() + mid_gate
                        p = (nnow_gate.sum(dim=1) + 0.01) / (con_pre_gate.sum(dim=1) + 0.1)
                        diversity_loss = (p.mul(p.log()) + (1 - p).mul((1 - p).log()))
                        if number == 0:
                            diversity_loss_total.append(diversity_loss)
                        else:
                            diversity_loss_total[i] += diversity_loss
                    number += 1

            else:
                gate3 = fre_gate_list.pop(0).cuda()
                gate3 = gate3.unsqueeze(0)
                gate3 = gate3.unsqueeze(-1)
                gate3 = gate3.unsqueeze(-1)
                out3 = gate3 * out3

        out3 += shortcut
        out3 = F.relu_(out3)
        return out3, gate_loss, diversity_loss_total, gate_list, pre_gate_list,fre_gate_list


class DeformBottleneckBlock(CNNBlockBase):
    """
    Similar to :class:`BottleneckBlock`, but with :paper:`deformable conv <deformconv>`
    in the 3x3 convolution.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            *,
            bottleneck_channels,
            stride=1,
            num_groups=1,
            norm="BN",
            stride_in_1x1=False,
            dilation=1,
            deform_modulated=False,
            deform_num_groups=1,
    ):
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
        )
        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem_Gate(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(self, gate=False, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gate = gate
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        if self.gate:
            self.gate1_fc1 = nn.Linear(in_channels, 16)
            self.gate1_fc2 = nn.Linear(16, out_channels)

        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x, gate_list_fre=None):
        gate_loss = 0
        gate_list = []
        out = self.conv1(x)
        if self.gate:
            if gate_list_fre == None:

                gate1 = F.adaptive_avg_pool2d(x, (1, 1))
                gate1 = gate1.view(-1, self.in_channels)
                gate1 = self.gate1_fc1(gate1)
                gate1 = F.relu(gate1)
                gate1 = self.gate1_fc2(gate1)
                noise1 = -torch.empty_like(gate1).exponential_().log()
                gate1 = noise1 + gate1
                gate1 = torch.sigmoid(gate1)
                gate_loss = gate1.sum() / self.out_channels
                gate_list.append(gate1.sum(dim=0))
                out = out.permute(2, 3, 0, 1)
                out = out * gate1
                out = out.permute(2, 3, 0, 1)
            else:
                gate1 = gate_list_fre.pop(0).cuda()
                out = out.permute(2, 3, 0, 1)
                out = out * gate1
                out = out.permute(2, 3, 0, 1)

        out = F.relu_(out)
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        if gate_list_fre == None:
            return out, gate_loss, gate_list
        else:
            return out, gate_loss, gate_list_fre


class ResNet_Gate(Backbone):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, cfg, stem, stages,gate=False, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super().__init__()
        self.fre_gate_input = cfg.MODEL.BACKBONE_FRE_GATE_INPUT
        self.gate_list_fre_name = cfg.MODEL.BACKBONE_GATELIST_FRE
        self.gate_list_pre_name = cfg.MODEL.BACKBONE_GATELIST_TRAIN
        if self.gate_list_fre_name == None:
            self.gate_list_fre = None
        else:
            self.gate_list_fre = torch.load(self.gate_list_fre_name,map_location = f'cuda:{dist.get_rank()}')
            self.gate_list_fre = self.gate_list_fre[int((gate-1)*len(self.gate_list_fre)/cfg.MODEL.NUM_BACKBONE):int(gate*len(self.gate_list_fre)/cfg.MODEL.NUM_BACKBONE)]
        if self.gate_list_pre_name == None:
            self.gate_list_pre = None
        else:
            if gate == cfg.MODEL.NUM_BACKBONE:
                self.gate_list_pre = torch.load(self.gate_list_pre_name,map_location = f'cuda:{dist.get_rank()}')
            else:
                self.gate_list_pre = None


        self.freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
        self.stem = stem
        self.num_classes = num_classes
        self.dis_name = cfg.MODEL.DISTILLATION_FEATURE

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.ModuleList(blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        dis_feature = {}
        gate_list_total = []
        gate_loss_total = 0
        diversity_loss_total = []
        if self.gate_list_fre == None:
            gate_list_fre = None
        else:
            gate_list_fre = copy.deepcopy(self.gate_list_fre)
        if self.gate_list_pre == None:
            gate_list_pre = None
        else:
            gate_list_pre = copy.deepcopy(self.gate_list_pre)
        if self.fre_gate_input == None:
            x, gate_loss, gate_stem = self.stem(x)
            gate_list_total = gate_list_total + gate_stem
            if "stem" in self._out_features:
                outputs["stem"] = x

            for name, stage in zip(self.stage_names, self.stages):
                for block in stage:
                    x, gate_loss, diversity_loss, gate_list, gate_list_pre,gate_list_fre = block(x, fre_gate_list=gate_list_fre,
                                                                                   fre_gate=self.fre_gate_input,pre_gate_list=gate_list_pre)
                    gate_loss_total = gate_loss_total + gate_loss
                    gate_list_total = gate_list_total + gate_list
                    if diversity_loss_total == [] :
                        diversity_loss_total = diversity_loss
                    else:
                        for i in range(len(diversity_loss_total)):
                            diversity_loss_total[i] += diversity_loss[i]

                if name in self._out_features:
                    outputs[name] = x
                if name in self.dis_name:
                    dis_feature[name] = x
        else:
            x, gate_loss, gate_stem = self.stem(x)
            if "stem" in self._out_features:
                outputs["stem"] = x

            i = 2

            for name, stage in zip(self.stage_names, self.stages):
                for block in stage:
                    if i > self.freeze_at:
                        x, gate_loss, diversity_loss, gate_list,_, gate_list_fre = block(x, fre_gate_list=gate_list_fre,
                                                                                   fre_gate=self.fre_gate_input,pre_gate_list=gate_list_pre)
                    else:
                        x, gate_loss, diversity_loss, gate_list, _ , _= block(x, fre_gate_list=None, fre_gate=None, pre_gate_list=None)

                if name in self._out_features:
                    outputs[name] = x
                if name in self.dis_name:
                    dis_feature[name] = x
                i = i + 1

        if len(gate_list_total) != 0:
            gate_loss_total /= len(gate_list_total)
            for diversity_loss in diversity_loss_total:
                diversity_loss /= len(gate_list_total)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x

        return outputs, gate_loss_total, gate_list_total, diversity_loss_total, dis_feature

    def dis_feature_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.dis_name
        }

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(
            block_class, num_blocks, first_stride=None, *, in_channels, out_channels, gate, pre_class, num_class,
            **kwargs
    ):
        """
        Create a list of blocks of the same type that forms one ResNet stage.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            first_stride (int): deprecated
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.

        Returns:
            list[nn.Module]: a list of block module.

        Examples:
        ::
            stages = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )

        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        if first_stride is not None:
            assert "stride" not in kwargs and "stride_per_block" not in kwargs
            kwargs["stride_per_block"] = [first_stride] + [1] * (num_blocks - 1)
            logger = logging.getLogger(__name__)
            logger.warning(
                "ResNet.make_stage(first_stride=) is deprecated!  "
                "Use 'stride_per_block' or 'stride' instead."
            )

        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(gate=gate, pre_class=pre_class, num_class=num_class, in_channels=in_channels,
                            out_channels=out_channels, **curr_kwargs)
            )
            in_channels = out_channels
        return blocks


ResNetBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
"""


def make_stage(*args, **kwargs):
    """
    Deprecated alias for backward compatibiltiy.
    """
    return ResNet_Gate.make_stage(*args, **kwargs)


@BACKBONE_REGISTRY.register()
def build_resnet_backbone_gate(cfg, input_shape, gate=False):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    # fmt: off
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    norm = cfg.MODEL.RESNETS.NORM
    pre_class = cfg.MODEL.PRE_CLASSES
    num_class = cfg.MODEL.NUM_CLASSES
    dataset_name = cfg.MODEL.DATASETNAME
    class_embedding = torch.load(f'class_embedding/{dataset_name}_classembedding.pt',map_location = f'cuda:{dist.get_rank()}')

    if freeze_at > 0:
        gate_stem = False
    else:
        gate_stem = gate

    stem = BasicStem_Gate(
        gate=gate_stem,
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [
        {"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features if f != "stem"
    ]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2

        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,

            "norm": norm,
        }
        # Use BasicBlock for R18 and R34.
        if (idx + 1) >= freeze_at:
            stage_kargs["gate"] = gate
        else:
            stage_kargs["gate"] = False
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock_Gate
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["pre_class"] = pre_class
            stage_kargs["num_class"] = num_class
            stage_kargs["class_embedding"] = class_embedding
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock_Gate

        blocks = ResNet_Gate.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet_Gate(cfg, stem, stages,gate=gate, out_features=out_features).freeze(freeze_at)
