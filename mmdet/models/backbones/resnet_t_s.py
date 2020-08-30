import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer, ConvModule
from ..builder import build_loss

import math
from IPython import embed


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, "Not implemented yet."
        assert gen_attention is None, "Not implemented yet."
        assert gcb is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert gcb is None or isinstance(gcb, dict)
        assert gen_attention is None or isinstance(gen_attention, dict)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.gcb = gcb
        self.with_gcb = gcb is not None
        self.gen_attention = gen_attention
        self.with_gen_attention = gen_attention is not None

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = dcn.get('fallback_on_stride', False)
            self.with_modulated_dcn = dcn.get('modulated', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert conv_cfg is None, 'conv_cfg must be None for DCN'
            self.deformable_groups = dcn.get('deformable_groups', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(
                planes,
                self.deformable_groups * offset_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation)
            self.conv2 = conv_op(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                deformable_groups=self.deformable_groups,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_gcb:
            gcb_inplanes = planes * self.expansion
            self.context_block = ContextBlock(inplanes=gcb_inplanes, **gcb)

        # gen_attention
        if self.with_gen_attention:
            self.gen_attention_block = GeneralizedAttention(
                planes, **gen_attention)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if not self.with_dcn:
                out = self.conv2(out)
            elif self.with_modulated_dcn:
                offset_mask = self.conv2_offset(out)
                offset = offset_mask[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.conv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.conv2(out, offset)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_gen_attention:
                out = self.gen_attention_block(out)
            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_gcb:
                out = self.context_block(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_rests_layer(block,
                     inplanes,
                     planes,
                     blocks,
                     stride=1,
                     dilation=1,
                     style='pytorch',
                     with_cp=False,
                     conv_cfg=None,
                     norm_cfg=dict(type='BN'),
                     dcn=None,
                     gcb=None,
                     gen_attention=None,
                     gen_attention_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gcb=gcb,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResTSNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 s_depth,
                 in_channels=3,
                 t_s_ratio=1,
                 spatial_ratio=1,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 pyramid_hint_loss=dict(type='MSELoss', loss_weight=1),
                 apply_block_wise_alignment=False,
                 freeze_teacher=False,
                 good_initial=False,
                 feature_adaption=False,
                 kernel_adaption=False,
                 conv_downsample=False,
                 train_mode=True,
                 constant_term=False,
                 pure_student_term=False,
                 bn_topk_selection=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 gcb=None,
                 stage_with_gcb=(False, False, False, False),
                 gen_attention=None,
                 stage_with_gen_attention=((), (), (), ()),
                 with_cp=False,
                 zero_init_residual=True,
                 rouse_student_point=0):
        super(ResTSNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.s_depth = s_depth
        self.t_s_ratio = t_s_ratio
        self.spatial_ratio = spatial_ratio
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.pyramid_hint_loss = build_loss(pyramid_hint_loss)
        self.apply_block_wise_alignment = apply_block_wise_alignment
        self.freeze_teacher = freeze_teacher
        self.frozen_stages = frozen_stages
        self.good_initial = good_initial
        self.feature_adaption = feature_adaption
        self.kernel_adaption = kernel_adaption
        self.conv_downsample = conv_downsample
        self.bn_topk_selection = bn_topk_selection
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.gen_attention = gen_attention
        self.gcb = gcb
        self.stage_with_gcb = stage_with_gcb
        if gcb is not None:
            assert len(stage_with_gcb) == num_stages
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.s_block, s_stage_blocks = self.arch_settings[s_depth]
        self.s_stage_blocks = s_stage_blocks[:num_stages]
        self.inplanes = 64
        self.rouse_student_point = rouse_student_point
        self.train_step = 0
        self.train_mode = train_mode
        self.constant_term = constant_term
        self.pure_student_term = pure_student_term

        self._make_stem_layer(in_channels)
        self._make_s_stem_layer(in_channels)

        self.res_layers = []
        self.s_res_layers = []
        self.align_layers = nn.ModuleList()
        # teacher net
        teacher_block_output_channel = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            planes = 64 * 2**i
            res_layer = make_rests_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i])
            teacher_block_output_channel.append(planes)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        self._freeze_stages()
        # student net
        # TODO: rewrite student layers;
        # current block1[0] layer input channel not fully pruned in same way
        self.inplanes = 64  # // self.t_s_ratio
        student_block_output_channel = []
        for j, num_blocks in enumerate(self.s_stage_blocks):
            stride = strides[j]
            dilation = dilations[j]
            dcn = self.dcn if self.stage_with_dcn[j] else None
            gcb = self.gcb if self.stage_with_gcb[j] else None
            planes = 64 * 2**j // self.t_s_ratio  # Prune the channel
            s_res_layer = make_rests_layer(
                self.s_block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[j])
            student_block_output_channel.append(planes)
            self.inplanes = planes * self.s_block.expansion
            s_layer_name = 's_layer{}'.format(j + 1)
            self.add_module(s_layer_name, s_res_layer)
            self.s_res_layers.append(s_layer_name)

        self.feat_dim = self.s_block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)
        # hint knowlege, align teacher and student
        self.inplanes = 64
        # TODO: Add to config file
        self.align_layers_output_channel_size = [256, 512, 1024, 2048]
        if self.apply_block_wise_alignment:
            for out_channel in self.align_layers_output_channel_size:
                input_channel = out_channel // self.t_s_ratio

                self.align_layers.append(
                    nn.Conv2d(input_channel, out_channel, 3, padding=1))
                # print("self.inplanes:{}".format(self.inplanes))
        if self.feature_adaption and self.conv_downsample:
            assert self.kernel_adaption == False

            self.adaption_channels = [256, 512, 1024, 2048]
            self.adaption_layers = nn.ModuleList()

            for adaption_channel in self.adaption_channels:
                self.adaption_layers.append(
                    nn.Conv2d(
                        adaption_channel,
                        adaption_channel // self.t_s_ratio,
                        3,
                        padding=1))
                '''
                self.adaption_layers.append(
                    nn.Conv2d(
                        adaption_channel,
                        adaption_channel // self.t_s_ratio,
                        1,
                        padding=0))
                '''
        if self.kernel_adaption:
            assert self.feature_adaption == False
            '''
            self.adaption_channels = [[[64, 64, 64, 64], [256, 64, 64],
                                       [256, 64, 64]],
                                      [[256, 128, 128, 256], [512, 128, 128],
                                       [512, 128, 128], [512, 128, 128]],
                                      [[512, 256, 256, 512], [1024, 256, 256],
                                       [1024, 256, 256], [1024, 256, 256],
                                       [1024, 256, 256], [1024, 256, 256]],
                                      [[1024, 512, 512, 1024],
                                       [2048, 512, 512], [2048, 512, 512]]]
            self.linear_channels = [[[64, 64, 256, 256], [64, 64, 256],
                                     [64, 64, 256]],
                                    [[128, 128, 512, 512], [128, 128, 512],
                                     [128, 128, 512], [128, 128, 512]],
                                    [[256, 256, 1024, 1024], [256, 256, 1024],
                                     [256, 256, 1024], [256, 256, 1024],
                                     [256, 256, 1024], [256, 256, 1024]],
                                    [[512, 512, 2048, 2048], [512, 512, 2048],
                                     [512, 512, 2048]]]
            '''
            '''
            self.adaption_channels = [[[64, -1, -1, -1], [256, -1, -1], [256, -1, -1]],
                                      [[256, -1, -1, -1], [512, -1, -1], [512, -1, -1], [512, -1, -1]],
                                      [[512, -1, -1, -1], [1024, -1, -1], [1024, -1, -1], [1024, -1, -1], [1024, -1, -1],
                                       [1024, -1, -1]], [[1024, -1, -1, -1], [2048, -1, -1], [2048, -1, -1]]]
            self.linear_channels = [[[64, -1, -1, -1], [64, -1, -1], [64, -1, -1]],
                                    [[128, -1, -1, -1], [128, -1, -1], [128, -1, -1], [128, -1, -1]],
                                    [[256, -1, -1, -1], [256, -1, -1], [256, -1, -1], [256, -1, -1], [256, -1, -1], [256, -1, -1]],
                                    [[512, -1, -1, -1], [512, -1, -1], [512, -1, -1]]]
            '''
            '''
            # one block alignment
            self.adaption_channels = [[[64, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1]],
                                      [[256, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, -1, -1]],
                                      [[512, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, -1, -1]],
                                      [[1024, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1]]]
            self.linear_channels = [[[64, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1]],
                                    [[128, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [-1, -1, -1]],
                                    [[256, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1]],
                                    [[512, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1]]]
            '''
            '''
            # two block alignment
            self.adaption_channels = [[[64, -1, -1, -1], [256, -1, -1],
                                       [-1, -1, -1]],
                                      [[256, -1, -1, -1], [512, -1, -1],
                                       [-1, -1, -1], [-1, -1, -1]],
                                      [[512, -1, -1, -1], [1024, -1, -1],
                                       [-1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, -1, -1]],
                                      [[1024, -1, -1, -1], [2048, -1, -1],
                                       [-1, -1, -1]]]
            self.linear_channels = [[[64, -1, -1, -1], [64, -1, -1],
                                     [-1, -1, -1]],
                                    [[128, -1, -1, -1], [128, -1, -1],
                                     [-1, -1, -1], [-1, -1, -1]],
                                    [[256, -1, -1, -1], [256, -1, -1],
                                     [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1]],
                                    [[512, -1, -1, -1], [512, -1, -1],
                                     [-1, -1, -1]]]
            '''
            '''
            # middle block alignment
            self.adaption_channels = [[[-1, 64, -1, -1], [-1, 64, -1],
                                       [-1, 64, -1]],
                                      [[-1, 128, -1, -1], [-1, 128, -1],
                                       [-1, 128, -1], [-1, 128, -1]],
                                      [[-1, 256, -1, -1], [-1, 256, -1],
                                       [-1, 256, -1], [-1, 256, -1],
                                       [-1, 256, -1], [-1, 256, -1]],
                                      [[-1, 512, -1, -1],
                                       [-1, 512, -1], [-1, 512, -1]]]
            self.linear_channels = [[[-1, 64, -1, -1], [-1, 64, -1],
                                     [-1, 64, -1]],
                                    [[-1, 128, -1, -1], [-1, 128, -1],
                                     [-1, 128, -1], [-1, 128, -1]],
                                    [[-1, 256, -1, -1], [-1, 256, -1],
                                     [-1, 256, -1], [-1, 256, -1],
                                     [-1, 256, -1], [-1, 256, -1]],
                                    [[-1, 512, -1, -1], [-1, 512, -1],
                                     [-1, 512, -1]]]
            '''
            '''
            # three block alignment
            self.adaption_channels = [[[64, -1, -1, -1], [256, -1, -1], [256, -1, -1]],
                                      [[256, -1, -1, -1], [512, -1, -1], [512, -1, -1], [-1, -1, -1]],
                                      [[512, -1, -1, -1], [1024, -1, -1], [1024, -1, -1], [-1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1]], [[1024, -1, -1, -1], [2048, -1, -1], [2048, -1, -1]]]
            self.linear_channels = [[[64, -1, -1, -1], [64, -1, -1], [64, -1, -1]],
                                    [[128, -1, -1, -1], [128, -1, -1], [128, -1, -1], [-1, -1, -1]],
                                    [[256, -1, -1, -1], [256, -1, -1], [256, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
                                    [[512, -1, -1, -1], [512, -1, -1], [512, -1, -1]]]
            '''
            # all block alignment
            '''
            self.adaption_channels = [[[64, -1, -1, -1], [256, -1, -1], [256, -1, -1]],
                                      [[256, -1, -1, -1], [512, -1, -1], [512, -1, -1], [512, -1, -1]],
                                      [[512, -1, -1, -1], [1024, -1, -1], [1024, -1, -1], [1024, -1, -1], [1024, -1, -1],
                                       [1024, -1, -1]], [[1024, -1, -1, -1], [2048, -1, -1], [2048, -1, -1]]]
            self.linear_channels = [[[64, -1, -1, -1], [64, -1, -1], [64, -1, -1]],
                                    [[128, -1, -1, -1], [128, -1, -1], [128, -1, -1], [128, -1, -1]],
                                    [[256, -1, -1, -1], [256, -1, -1], [256, -1, -1], [256, -1, -1], [256, -1, -1], [256, -1, -1]],
                                    [[512, -1, -1, -1], [512, -1, -1], [512, -1, -1]]]
            '''
            '''
            # shallow block2
            self.adaption_channels = [[[64, 64, -1, -1], [256, 64, -1],
                                       [256, 64, -1]],
                                      [[256, 128, -1, -1], [512, 128, -1],
                                       [512, 128, -1], [512, 128, -1]],
                                      [[512, 256, -1, -1], [1024, 256, -1],
                                       [1024, 256, -1], [1024, 256, -1],
                                       [1024, 256, -1], [1024, 256, -1]],
                                      [[1024, 512, -1, -1], [2048, 512, -1],
                                       [2048, 512, -1]]]
            self.linear_channels = [[[64, 64, -1, -1], [64, 64, -1],
                                     [64, 64, -1]],
                                    [[128, 128, -1, -1], [128, 128, -1],
                                     [128, 128, -1], [128, 128, -1]],
                                    [[256, 256, -1, -1], [256, 256, -1],
                                     [256, 256, -1], [256, 256, -1],
                                     [256, 256, -1], [256, 256, -1]],
                                    [[512, 512, -1, -1], [512, 512, -1],
                                     [512, 512, -1]]]
            '''
            '''
            # deep block1
            self.adaption_channels = [[[-1, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, 64]],
                                      [[-1, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, -1, 128]],
                                      [[-1, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, -1, 256]],
                                      [[-1, -1, -1, -1],
                                       [-1, -1, -1], [-1, -1, 512]]]
            self.linear_channels = [[[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, 256]],
                                    [[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [-1, -1, 512]],
                                    [[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [-1, -1, 1024]],
                                    [[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, 2048]]]
            '''
            
            # deep block2
            self.adaption_channels = [[[-1, -1, -1, -1], [-1, -1, -1],
                                       [-1, 64, 64]],
                                      [[-1, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, 128, 128]],
                                      [[-1, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, 256, 256]],
                                      [[-1, -1, -1, -1],
                                       [-1, -1, -1], [-1, 512, 512]]]
            self.linear_channels = [[[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, 64, 256]],
                                    [[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [-1, 128, 512]],
                                    [[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [-1, 256, 1024]],
                                    [[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, 512, 2048]]]
            
            '''
            # deep block3
            self.adaption_channels = [[[-1, -1, -1, -1], [-1, -1, -1],
                                       [256, 64, 64]],
                                      [[-1, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [512, 128, 128]],
                                      [[-1, -1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [-1, -1, -1],
                                       [-1, -1, -1], [1024, 256, 256]],
                                      [[-1, -1, -1, -1],
                                       [-1, -1, -1], [2048, 512, 512]]]
            self.linear_channels = [[[-1, -1, -1, -1], [-1, -1, -1],
                                     [64, 64, 256]],
                                    [[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [128, 128, 512]],
                                    [[-1, -1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [-1, -1, -1],
                                     [-1, -1, -1], [256, 256, 1024]],
                                    [[-1, -1, -1, -1], [-1, -1, -1],
                                     [512, 512, 2048]]]
            '''

            self.downsample_layers_group = nn.ModuleList()
            self.adaption_layers_group = nn.ModuleList()
            self.linear_layers_group = nn.ModuleList()
            self.conv1_adaption_3d = nn.Conv3d(
                64,
                64 // self.t_s_ratio,
                kernel_size=(3 // self.t_s_ratio, 1, 1),
                padding=0)
            '''
            self.conv1_linear = nn.Linear(64, 32, bias=False)
            for i in range(len(self.adaption_channels)):
                adaption_blocks = nn.ModuleList()
                linear_blocks = nn.ModuleList()

                for j in range(len(self.adaption_channels[i])):
                    adaption_layers = nn.ModuleList()
                    for adaption_channel in self.adaption_channels[i][j]:
                        adaption_layers.append(
                            nn.Conv2d(
                                adaption_channel,
                                adaption_channel // self.t_s_ratio,
                                1,
                                padding=0))
                    adaption_blocks.append(adaption_layers)
                self.adaption_layers_group.append(adaption_blocks)

                for k in range(len(self.linear_channels[i])):
                    linear_layers = nn.ModuleList()
                    for linear_channel in self.linear_channels[i][k]:
                        linear_layers.append(
                            nn.Linear(
                                linear_channel,
                                linear_channel // self.t_s_ratio,
                                bias=False))
                    linear_blocks.append(linear_layers)
                self.linear_layers_group.append(linear_blocks)
            '''
            # 3d conv
            for i in range(len(self.adaption_channels)):
                downsample_blocks = nn.ModuleList()
                adaption_blocks = nn.ModuleList()

                for j in range(len(self.adaption_channels[i])):
                    downsample_layers = nn.ModuleList()
                    adaption_layers = nn.ModuleList()

                    for linear_channel, adaption_channel in zip(
                            self.linear_channels[i][j],
                            self.adaption_channels[i][j]):
                        if linear_channel != -1 and adaption_channel != -1:
                            if i == 0 and j == 0: # FIXME: test for merge first layer before blocks
                                downsample_layers.append(
                                    nn.Conv2d(
                                        adaption_channel,
                                        adaption_channel,
                                        kernel_size=3,
                                        padding=1))
                            else:
                                downsample_layers.append(
                                    nn.Conv2d(
                                        adaption_channel,
                                        adaption_channel // self.t_s_ratio,
                                        kernel_size=3,
                                        padding=1))
                            adaption_layers.append(
                                nn.Conv3d(
                                    linear_channel,
                                    linear_channel // self.t_s_ratio,
                                    kernel_size=(1, 5,
                                                 5),  #kernel_size=(3, 3, 3),
                                    padding=(0, 2, 2)))  #padding=(1, 1, 1)))
                            '''
                            adaption_layers.append(
                                nn.Conv3d(
                                    linear_channel,
                                    linear_channel // self.t_s_ratio,
                                    kernel_size=(
                                        adaption_channel // self.t_s_ratio + 1,
                                        3, 3),
                                    padding=(0, 1, 1)))
                            '''
                        else:
                            downsample_layers.append(nn.ModuleList())
                            adaption_layers.append(nn.ModuleList())

                    downsample_blocks.append(downsample_layers)
                    adaption_blocks.append(adaption_layers)

                self.downsample_layers_group.append(downsample_blocks)
                self.adaption_layers_group.append(adaption_blocks)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def s_norm1(self):
        return getattr(self, self.s_norm1_name)

    def _make_stem_layer(self, in_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_s_stem_layer(self, in_channels):
        self.s_conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64 // self.t_s_ratio,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.s_norm1_name, s_norm1 = build_norm_layer(
            self.norm_cfg, 64 // self.t_s_ratio, postfix=2)
        self.add_module(self.s_norm1_name, s_norm1)
        self.s_relu = nn.ReLU(inplace=True)
        self.s_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.freeze_teacher:
            assert self.frozen_stages == 4
        else:
            assert self.frozen_stages == 1

        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def adapt_kernel_train(self, s_x, j, l, t_layer, s_layer):
        identity = s_x
        '''
        linear_layers = self.linear_layers_group[j][l]
        '''

        # conv
        t_layer_conv1_data = t_layer.conv1.weight.detach()
        t_layer_conv2_data = t_layer.conv2.weight.detach()
        t_layer_conv3_data = t_layer.conv3.weight.detach()

        downsamples_layers = self.downsample_layers_group[j][l]
        adaption_layers = self.adaption_layers_group[j][l]
        if adaption_layers[0] and downsamples_layers[0]:
            t_layer_conv1_data = downsamples_layers[0](t_layer_conv1_data)
            # match the adaption kernel size for adaption
            t_layer_conv1_data = torch.squeeze(
                adaption_layers[0](torch.unsqueeze(t_layer_conv1_data,
                                                   axis=0)), 0)
            # NOTE: Manually apply convolution on student features
            # TODO: F.relu cannot be inplace, figure out why
            s_out = F.conv2d(s_x, t_layer_conv1_data, stride=(1, 1))
        else:
            s_out = s_layer.conv1(s_x)

        s_out = s_layer.bn1(s_out)
        s_out = F.relu(s_out)

        if adaption_layers[1] and downsamples_layers[1]:
            t_layer_conv2_data = downsamples_layers[1](t_layer_conv2_data)
            # match the adaption kernel size for adaption
            t_layer_conv2_data = torch.squeeze(
                adaption_layers[1](torch.unsqueeze(t_layer_conv2_data,
                                                   axis=0)), 0)
            # NOTE: downsample first block from second bottleneck
            if j >= 1 and l == 0:
                s_out = F.conv2d(
                    s_out, t_layer_conv2_data, stride=(2, 2), padding=(1, 1))
            else:
                s_out = F.conv2d(
                    s_out, t_layer_conv2_data, stride=(1, 1), padding=(1, 1))
        else:
            s_out = s_layer.conv2(s_out)

        s_out = s_layer.bn2(s_out)
        s_out = F.relu(s_out)

        if adaption_layers[2] and downsamples_layers[2]:
            t_layer_conv3_data = downsamples_layers[2](t_layer_conv3_data)
            # match the adaption kernel size for adaption
            t_layer_conv3_data = torch.squeeze(
                adaption_layers[2](torch.unsqueeze(t_layer_conv3_data,
                                                   axis=0)), 0)
            s_out = F.conv2d(s_out, t_layer_conv3_data, stride=(1, 1))
        else:
            s_out = s_layer.conv3(s_out)

        s_out = s_layer.bn3(s_out)

        if t_layer.downsample is not None:
            t_layer_downsample_conv_data = t_layer.downsample[0].weight.data
            if adaption_layers[3] and downsamples_layers[3]:
                t_layer_downsample_conv_data = downsamples_layers[3](
                    t_layer_downsample_conv_data)
                # match the adaption kernel size for adaption
                t_layer_downsample_conv_data = torch.squeeze(
                    adaption_layers[3](torch.unsqueeze(
                        t_layer_downsample_conv_data, axis=0)), 0)
                if j >= 1:
                    identity = F.conv2d(
                        s_x, t_layer_downsample_conv_data, stride=(2, 2))
                else:
                    identity = F.conv2d(
                        s_x, t_layer_downsample_conv_data, stride=(1, 1))
            else:
                identity = s_layer.downsample[0](s_x)

            identity = s_layer.downsample[1](identity)

        s_out += identity
        s_out = F.relu(s_out)

        return s_out

    def adapt_kernel_inference(self, j, l, t_layer, s_layer):
        '''
        linear_layers = self.linear_layers_group[j][l]
        '''
        # conv
        t_layer_conv1_data = t_layer.conv1.weight.detach()
        t_layer_conv2_data = t_layer.conv2.weight.detach()
        t_layer_conv3_data = t_layer.conv3.weight.detach()

        s_conv1_weight = torch.squeeze(
            self.conv1_adaption_3d(torch.unsqueeze(self.conv1.weight.data, 0)),
            0)
        self.s_conv1.weight = torch.nn.Parameter(s_conv1_weight)
        downsamples_layers = self.downsample_layers_group[j][l]
        adaption_layers = self.adaption_layers_group[j][l]

        if adaption_layers[0] and downsamples_layers[0]:
            t_layer_conv1_data = downsamples_layers[0](t_layer_conv1_data)
            # match the adaption kernel size for adaption
            t_layer_conv1_data = torch.squeeze(
                adaption_layers[0](torch.unsqueeze(t_layer_conv1_data,
                                                   axis=0)), 0)
            s_layer.conv1.weight = torch.nn.Parameter(t_layer_conv1_data)

        if adaption_layers[1] and downsamples_layers[1]:
            t_layer_conv2_data = downsamples_layers[0](t_layer_conv2_data)
            # match the adaption kernel size for adaption
            t_layer_conv2_data = torch.squeeze(
                adaption_layers[1](torch.unsqueeze(t_layer_conv2_data,
                                                   axis=0)), 0)
            s_layer.conv2.weight = torch.nn.Parameter(t_layer_conv2_data)

        if adaption_layers[2] and downsamples_layers[2]:
            t_layer_conv3_data = downsamples_layers[2](t_layer_conv3_data)
            # match the adaption kernel size for adaption
            t_layer_conv3_data = torch.squeeze(
                adaption_layers[2](torch.unsqueeze(t_layer_conv3_data,
                                                   axis=0)), 0)
            s_layer.conv3.weight = torch.nn.Parameter(t_layer_conv3_data)

        if s_layer.downsample is not None:
            t_layer_downsample_conv_data = t_layer.downsample[0].weight.data

            if adaption_layers[3] and downsamples_layers[3]:
                t_layer_downsample_conv_data = downsamples_layers[3](
                    t_layer_downsample_conv_data)
                # match the adaption kernel size for adaption
                t_layer_downsample_conv_data = torch.squeeze(
                    adaption_layers[3](torch.unsqueeze(
                        t_layer_downsample_conv_data, axis=0)), 0)
                s_layer.downsample[0].weight = torch.nn.Parameter(
                    t_layer_downsample_conv_data)

    def copy_backbone(self):
        factor = 0.5
        # stem layer
        self.s_conv1.weight.data = factor * self.s_conv1.weight.data + (
            1 - factor) * F.interpolate(
                self.conv1.weight.data.permute(2, 3, 0, 1).detach(),
                size=self.s_conv1.weight.shape[:2],
                mode='bilinear').permute(2, 3, 0, 1)

        for m in self.modules():
            if hasattr(m, 's_layer1'):
                t_bottleneck_list = [m.layer1, m.layer2, m.layer3, m.layer4]
                s_bottleneck_list = [
                    m.s_layer1, m.s_layer2, m.s_layer3, m.s_layer4
                ]
                # t_bottleneck_list = [t_layers1]
                # s_bottleneck_list = [s_layers1]
                for t_layers, s_layers in zip(t_bottleneck_list,
                                              s_bottleneck_list):
                    for t_layer, s_layer in zip(t_layers, s_layers):
                        # conv
                        t_layer_conv1_data = t_layer.conv1.weight.data.permute(
                            2, 3, 0, 1).detach()
                        s_layer.conv1.weight.data = factor * s_layer.conv1.weight.data + (
                            1 - factor) * F.interpolate(
                                t_layer_conv1_data,
                                size=s_layer.conv1.weight.shape[:2],
                                mode='bilinear').permute(2, 3, 0, 1)
                        t_layer_conv2_data = t_layer.conv2.weight.data.permute(
                            2, 3, 0, 1).detach()
                        s_layer.conv2.weight.data = factor * s_layer.conv2.weight.data + (
                            1 - factor) * F.interpolate(
                                t_layer_conv2_data,
                                size=s_layer.conv2.weight.shape[:2],
                                mode='bilinear').permute(2, 3, 0, 1)
                        t_layer_conv3_data = t_layer.conv3.weight.data.permute(
                            2, 3, 0, 1).detach()
                        s_layer.conv3.weight.data = factor * s_layer.conv3.weight.data + (
                            1 - factor) * F.interpolate(
                                t_layer_conv3_data,
                                size=s_layer.conv3.weight.shape[:2],
                                mode='bilinear').permute(2, 3, 0, 1)

                        if t_layer.downsample is not None:
                            # donwsample
                            t_layer_downsample_conv_data = t_layer.downsample[
                                0].weight.data.permute(2, 3, 0, 1)
                            s_layer.downsample[
                                0].weight.data = factor * s_layer.downsample[
                                    0].weight.data + (
                                        1 - factor) * F.interpolate(
                                            t_layer_downsample_conv_data,
                                            size=s_layer.downsample[0].weight.
                                            shape[:2],
                                            mode='bilinear').permute(
                                                2, 3, 0, 1)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

            if self.good_initial:
                self.copy_backbone()
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

        if self.feature_adaption and self.conv_downsample:
            for m in self.adaption_layers:
                normal_init(m, std=0.01)

    def forward(self, x):
        # Adapt kernel once at inference
        if not self.train_mode and self.kernel_adaption and self.train_step == 0:
            for j, s_layer_name in enumerate(self.s_res_layers):
                s_layer_name = self.s_res_layers[j]
                t_layer_name = self.res_layers[j]
                s_bottlenecks = getattr(self, s_layer_name)
                t_bottlenecks = getattr(self, t_layer_name)

                for l, (t_layer, s_layer) in enumerate(
                        zip(t_bottlenecks, s_bottlenecks)):
                    self.adapt_kernel_inference(j, l, t_layer, s_layer)

        self.train_step += 1
        # '''
        if self.spatial_ratio != 1:
            s_x = F.interpolate(x, scale_factor=1 / self.spatial_ratio)
        else:
            s_x = x
        # '''
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        '''
        if self.kernel_adaption and self.train_mode:
            # s_conv1_weight = self.conv1_linear(
            #     self.conv1.weight.data.permute(1, 2, 3,
            #                                    0)).permute(3, 0, 1, 2)
            s_conv1_weight = torch.squeeze(
                self.conv1_adaption_3d(
                    torch.unsqueeze(self.conv1.weight.data, 0)), 0)

            s_x = F.conv2d(s_x, s_conv1_weight, stride=(2, 2), padding=(3, 3))
        else:
            s_x = self.s_conv1(s_x)

        s_x = self.s_norm1(s_x)
        s_x = self.s_relu(s_x)
        s_x = self.s_maxpool(s_x)
        '''

        if self.spatial_ratio != 1:
            s_x = F.interpolate(x, scale_factor=1 / self.spatial_ratio)
        else:
            s_x = x

        if self.pure_student_term:
            pure_s_x = s_x
            s_pure_outs = []

        inputs = []
        outs = []
        s_outs = []
        # hint_losses = []
        block_distill_pairs = []

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            '''
            if self.feature_adaption:
                inputs.append(x)
            '''
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        # student net
        for j, s_layer_name in enumerate(self.s_res_layers):
            s_res_layer = getattr(self, s_layer_name)

            # FIXME: no gradient update in adaption layers
            if self.kernel_adaption and self.train_mode:
                # no 'copy' bn yet
                # with torch.no_grad():
                s_layer_name = self.s_res_layers[j]
                t_layer_name = self.res_layers[j]
                s_bottlenecks = getattr(self, s_layer_name)
                t_bottlenecks = getattr(self, t_layer_name)

                for l, (t_layer, s_layer) in enumerate(
                        zip(t_bottlenecks, s_bottlenecks)):
                    s_x = self.adapt_kernel_train(s_x, j, l, t_layer, s_layer)
            else:
                s_x = s_res_layer(s_x)
            '''
            if self.feature_adaption and self.train_mode:
                adaption_factor = 0.5

                s_x = s_res_layer(s_x)

                if self.pure_student_term:
                    pure_s_x = s_res_layer(pure_s_x)

                if self.conv_downsample:
                    # x_detached = inputs[j].detach()
                    x_detached = outs[j].detach()
                    x_detached_adapted = self.adaption_layers[j](x_detached)

                    # align to teacher network and get the loss

                    s_x = adaption_factor * s_x + (
                        1 - adaption_factor) * x_detached_adapted

                    if self.apply_block_wise_alignment:
                        block_distill_pairs.append([s_x, x_detached_adapted])
                else:
                    x_detached = inputs[j].permute(2, 3, 0, 1).detach()
                    s_x = adaption_factor * s_x + (
                        1 - adaption_factor) * F.interpolate(
                            x_detached, size=s_x.shape[:2],
                            mode='bilinear').permute(2, 3, 0, 1)

            else:
                s_x = s_res_layer(s_x)
                _, _, feature_w, feature_h = s_x.shape
            '''
            if j in self.out_indices:
                s_outs.append(s_x)
                if self.pure_student_term:
                    s_pure_outs.append(pure_s_x)

        if self.apply_block_wise_alignment:
            return tuple(outs), tuple(s_outs), tuple(block_distill_pairs)
        elif self.pure_student_term:
            return tuple(outs), tuple(s_outs), tuple(s_pure_outs)
        else:
            return tuple(outs), tuple(s_outs)

    def train(self, mode=True):
        super(ResTSNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
