import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..builder import build_loss
from ..registry import NECKS
from ..utils import ConvModule
from mmcv.cnn import normal_init

from IPython import embed


@NECKS.register_module
class FPNTS(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            s_in_channels,
            s_out_channels,
            num_outs,
            start_level=0,
            t_s_ratio=4,
            end_level=-1,
            add_extra_convs=False,
            extra_convs_on_inputs=True,
            relu_before_extra_convs=False,
            pure_student_term=False,
            apply_block_wise_alignment=False,
            kernel_meta_learner=False,
            copy_teacher_fpn=False,
            no_norm_on_lateral=False,
            freeze_teacher=False,
            conv_cfg=None,
            norm_cfg=None,
            activation=None,
            rouse_student_point=0,
            pyramid_kernel_loss=dict(type='MSELoss', loss_weight=1),
    ):
        super(FPNTS, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.s_in_channels = s_in_channels
        self.out_channels = out_channels
        self.s_out_channels = s_out_channels
        self.t_s_ratio = t_s_ratio
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.pure_student_term = pure_student_term
        self.apply_block_wise_alignment = apply_block_wise_alignment
        self.kernel_meta_learner = kernel_meta_learner
        self.copy_teacher_fpn = copy_teacher_fpn
        self.no_norm_on_lateral = no_norm_on_lateral
        self.freeze_teacher = freeze_teacher
        self.fp16_enabled = False
        self.rouse_student_point = rouse_student_point
        self.pyramid_kernel_loss = build_loss(pyramid_kernel_loss)
        self.train_step = 0

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        # Teacher Net
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        # student Net
        self.s_lateral_convs = nn.ModuleList()
        self.s_fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                s_in_channels[i],
                s_out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                s_out_channels,
                s_out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.s_lateral_convs.append(l_conv)
            self.s_fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    s_in_channels = self.s_in_channels[self.backbone_end_level
                                                       - 1]
                else:
                    s_in_channels = s_out_channels
                extra_fpn_conv = ConvModule(
                    s_in_channels,
                    s_out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.s_fpn_convs.append(extra_fpn_conv)

        if self.copy_teacher_fpn:
            # NOTE: create a copy of teacher fpn
            self.t_copy_lateral_convs = nn.ModuleList()
            self.t_copy_fpn_convs = nn.ModuleList()
            self.align_t_copy_fpn_conv = nn.ModuleList()
            self.align_t_copy_back_conv = nn.ModuleList()
            self.align_t_copy_back_conv.append(
                ConvModule(
                    self.out_channels,
                    self.s_out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False))

            for i in range(self.start_level, self.backbone_end_level):
                l_conv = ConvModule(
                    self.in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    activation=self.activation,
                    inplace=False)
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)

                self.t_copy_lateral_convs.append(l_conv)
                self.t_copy_fpn_convs.append(fpn_conv)
                self.align_t_copy_fpn_conv.append(
                    ConvModule(
                        self.s_in_channels[i - self.start_level],
                        self.in_channels[i - self.start_level],
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False))
            # add extra conv layers (e.g., RetinaNet)
            extra_levels = num_outs - self.backbone_end_level + self.start_level
            if add_extra_convs and extra_levels >= 1:
                for i in range(extra_levels):
                    if i == 0 and self.extra_convs_on_inputs:
                        in_channels = self.in_channels[self.backbone_end_level
                                                       - 1]
                    else:
                        in_channels = out_channels
                    extra_fpn_conv = ConvModule(
                        in_channels,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False)
                    self.t_copy_fpn_convs.append(extra_fpn_conv)
            self.align_t_copy_fpn_conv.append(
                nn.Conv2d(
                    self.s_in_channels[-1], self.in_channels[-1], 3,
                    padding=1))

        if self.kernel_meta_learner:
            # NOTE: Try to learn the kernel (do conv on teacher student pyramid convs)
            # 128 * 128 --> 256 * 256
            # TODO: Refactor later
            self.kernel_convs = nn.ModuleList()
            for _ in range(5):
                pyramid_kernel_conv = nn.ModuleList()
                # pyramid_kernel_conv.append(nn.Conv2d(256, 128, 3,
                #                                      padding=1))  # channel
                # pyramid_kernel_conv.append(nn.Conv2d(256, 128, 3,
                #                                      padding=1))  # kernel nums
                pyramid_kernel_conv.append(
                    nn.Linear(256 * 3 * 3, 128 * 3 * 3, bias=False))  # channel
                pyramid_kernel_conv.append(nn.Linear(
                    256, 128, bias=False))  # kernel nums
                pyramid_kernel_conv.append(nn.Linear(256, 128, bias=False))
                self.kernel_convs.append(pyramid_kernel_conv)

    # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        if self.freeze_teacher:
            self._freeze_teacher_layers()
        if self.copy_teacher_fpn:
            self._copy_freeze_fpn()
        if self.kernel_meta_learner:
            for kernel_conv in self.kernel_convs:
                for m in kernel_conv:
                    # xavier_init(m, distribution='uniform')
                    normal_init(m, std=0.01)

    def _freeze_teacher_layers(self):
        for fpn_conv in self.fpn_convs:
            fpn_conv.eval()
            for param in fpn_conv.parameters():
                param.requires_grad = False

        for lateral_conv in self.lateral_convs:
            lateral_conv.eval()
            for param in lateral_conv.parameters():
                param.requires_grad = False

    def _copy_freeze_fpn(self):
        for fpn_conv, origin_fpn_conv in zip(self.t_copy_fpn_convs,
                                             self.fpn_convs):
            fpn_conv.eval()
            for param, origin_param in zip(fpn_conv.parameters(),
                                           origin_fpn_conv.parameters()):
                param = origin_param.detach()

        for lateral_conv, origin_lateral_conv in zip(self.t_copy_lateral_convs,
                                                     self.lateral_convs):
            lateral_conv.eval()
            for param, origin_param in zip(lateral_conv.parameters(),
                                           origin_lateral_conv.parameters()):
                param = origin_param.detach()

    def copy_pyramid(self):
        for s_fpn_conv, t_fpn_conv in zip(self.s_fpn_convs, self.fpn_convs):
            t_layer_conv_data = t_fpn_conv.conv.weight.data.permute(
                2, 3, 0, 1).detach()
            s_fpn_conv.conv.weight.data.copy_(
                F.interpolate(
                    t_layer_conv_data,
                    size=s_fpn_conv.conv.weight.shape[:2],
                    mode='bilinear').permute(2, 3, 0, 1))

        for s_lateral_conv, t_lateral_conv in zip(self.s_lateral_convs,
                                                  self.lateral_convs):
            t_lateral_layer_conv_data = t_lateral_conv.conv.weight.data.permute(
                2, 3, 0, 1).detach()
            s_lateral_conv.conv.weight.data.copy_(
                F.interpolate(
                    t_lateral_layer_conv_data,
                    size=s_lateral_conv.conv.weight.shape[:2],
                    mode='bilinear').permute(2, 3, 0, 1))

    @auto_fp16()
    def forward(self, inputs):
        self.train_step += 1
        '''
        if self.rouse_student_point == self.train_step:
            self.copy_pyramid()
        '''
        if self.kernel_meta_learner:
            kernel_loss_tuple = tuple()
            for s_fpn_conv, fpn_conv, kernel_conv in zip(
                    self.s_fpn_convs, self.fpn_convs, self.kernel_convs):
                # 256 * 256 -> 128 * 128
                # s_conv_kernel_weights = s_fpn_conv.conv.weight

                t_conv_kernel_weights = fpn_conv.conv.weight.detach()
                # 256 * (256 * 3 * 3) -> 256 * (128 * 3 * 3)
                s_conv_kernel_weights = kernel_conv[0](
                    t_conv_kernel_weights.view(256, -1)).permute(1, 0)
                s_conv_kernel_weights = kernel_conv[1](
                    s_conv_kernel_weights).permute(1, 0).view(128, -1, 3, 3)

                s_fpn_conv.conv.weight = torch.nn.Parameter(
                    s_conv_kernel_weights.clone())
                s_fpn_conv.conv.bias = torch.nn.Parameter(kernel_conv[2](
                    fpn_conv.conv.bias.detach()))
                '''
                pyramid_kernel_loss = self.pyramid_kernel_loss(
                    s_conv_kernel_weights, t_conv_kernel_weights.detach())
                kernel_loss_tuple += tuple([pyramid_kernel_loss])
                '''

        # Teacher Net
        t_outs = self.single_forward(inputs[0], self.fpn_convs,
                                     self.lateral_convs)
        # Student Net
        s_outs = self.single_forward(inputs[1], self.s_fpn_convs,
                                     self.s_lateral_convs)
        if self.pure_student_term:
            pure_s_outs = self.single_forward(inputs[2], self.s_fpn_convs,
                                              self.s_lateral_convs)

        if self.copy_teacher_fpn:
            aligned_inputs = tuple()
            aligned_outputs = tuple()

            for i, s_input in enumerate(inputs[1]):
                aligned_inputs += tuple(
                    [self.align_t_copy_fpn_conv[i](s_input)])
            # NOTE: align to copied teacher fpn layers
            sharing_outs = self.single_forward(aligned_inputs,
                                               self.t_copy_fpn_convs,
                                               self.t_copy_lateral_convs)
            for sharing_out in sharing_outs:
                aligned_outputs += tuple(
                    [self.align_t_copy_back_conv[0](sharing_out)])

        if self.apply_block_wise_alignment:
            # push hint loss to head
            return tuple(t_outs), tuple(s_outs), tuple(inputs[0]), tuple(
                inputs[1]), tuple(inputs[2])
        elif self.pure_student_term:
            # pure student fpn fetures <-align-> teacher fpn features
            return tuple(t_outs), tuple(s_outs), tuple(inputs[0]), tuple(
                inputs[1]), tuple(pure_s_outs)
        elif self.copy_teacher_fpn:
            return tuple(t_outs), tuple(s_outs), tuple(inputs[0]), tuple(
                inputs[1]), tuple(aligned_outputs)
        else:
            return tuple(t_outs), tuple(s_outs), tuple(inputs[0]), tuple(
                inputs[1])

    @auto_fp16()
    def single_forward(self, single_input, fpn_convs, lateral_convs):
        assert len(single_input) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(single_input[i + self.start_level])
            for i, lateral_conv in enumerate(lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = single_input[self.backbone_end_level - 1]
                    outs.append(fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(fpn_convs[i](outs[-1]))
        return tuple(outs)
