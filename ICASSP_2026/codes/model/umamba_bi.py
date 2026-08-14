import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Sequence, Tuple, List, Type
from torch.cuda.amp import autocast
from mamba_ssm import Mamba
from einops import rearrange

from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers import get_act_layer
from monai.networks.layers.convutils import same_padding

from .attentiondenseunet import (
    ConvolutionUnit, PixelShuffle, PixelUnshuffle, PixelUpsample, PixelDownsample,
    UpsampleLayer, DownsampleLayer, CBAM, default_init_weights, ChannelAttention, SpatialAttention
)

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
import math

class UpsampleLayerUMamba(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
        ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim,    # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        # Backward Mamba for bidirectional processing
        self.mamba_backward = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
        )
        self.channel_token = channel_token # whether to use channel as tokens

        self.bi_weight = nn.Parameter(torch.tensor(0.5))

    # def forward_patch_token(self, x):
    #     B, d_model = x.shape[:2]
    #     assert d_model == self.dim
    #     n_tokens = x.shape[2:].numel()
    #     img_dims = x.shape[2:]
        
    #     # Forward direction
    #     x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
    #     x_norm = self.norm(x_flat)
    #     x_mamba_forward = self.mamba(x_norm)
        
    #     # Backward direction
    #     x_flat_backward = torch.flip(x_flat, dims=[1])  # Reverse sequence
    #     x_norm_backward = self.norm(x_flat_backward) 
    #     x_mamba_backward = self.mamba_backward(x_norm_backward)

    #     # Flip the OUTPUT back to align positions
    #     x_mamba_backward = torch.flip(x_mamba_backward, dims=[1])
        
    #     # combine - same voxel positions
    #     # x_mamba = (x_mamba_forward + x_mamba_backward) / 2
    #     w = torch.sigmoid(self.bi_weight)
    #     x_mamba = w * x_mamba_forward + (1 - w) * x_mamba_backward
        
    #     out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
    #     return out

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        # Reorder to (Y,X,Z) so adjacent Z slices are 1 step apart in sequence
        x_perm = x.permute(0, 1, 3, 4, 2)          # B,C,Z,Y,X → B,C,Y,X,Z

        # Forward direction
        x_flat = x_perm.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba_forward = self.mamba(x_norm)

        # Backward direction
        x_flat_backward = torch.flip(x_flat, dims=[1])
        x_norm_backward = self.norm(x_flat_backward)
        x_mamba_backward = self.mamba_backward(x_norm_backward)
        x_mamba_backward = torch.flip(x_mamba_backward, dims=[1])

        # Learned weighted fusion
        w = torch.sigmoid(self.bi_weight)
        x_mamba = w * x_mamba_forward + (1 - w) * x_mamba_backward

        # Reshape back and restore original Z,Y,X order
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *x_perm.shape[2:])
        out = out.permute(0, 1, 4, 2, 3)            # B,C,Y,X,Z → B,C,Z,Y,X
        return out
        
    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        
        # Forward direction (original)
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba_forward = self.mamba(x_norm)
        
        # Backward direction
        x_flat_backward = torch.flip(x_flat, dims=[2])  # Reverse along spatial dim
        x_norm_backward = self.norm(x_flat_backward)
        x_mamba_backward = self.mamba_backward(x_norm_backward) 
        x_mamba_backward = torch.flip(x_mamba_backward, dims=[2])  # Flip back
        
        # Combine both directions
        x_mamba = (x_mamba_forward + x_mamba_backward) / 2
        
        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out


class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
        ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)
    

class ResidualMambaEncoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i // j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True
            

        print(f"feature_map_sizes: {feature_map_sizes}")
        print(f"do_channel_token: {do_channel_token}")

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op = conv_op,
                input_channels = input_channels,
                output_channels = stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            ), 
            *[
                BasicBlockD(
                    conv_op = conv_op,
                    input_channels = stem_channels,
                    output_channels = stem_channels,
                    kernel_size = kernel_sizes[0],
                    stride = 1,
                    conv_bias = conv_bias,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[0] - 1)
            ]
        )

        input_channels = stem_channels

        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op = conv_op,
                    norm_op = norm_op,
                    norm_op_kwargs = norm_op_kwargs,
                    input_channels = input_channels,
                    output_channels = features_per_stage[s],
                    kernel_size = kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin = nonlin,
                    nonlin_kwargs = nonlin_kwargs
                ),
                *[
                    BasicBlockD(
                        conv_op = conv_op,
                        input_channels = features_per_stage[s],
                        output_channels = features_per_stage[s],
                        kernel_size = kernel_sizes[s],
                        stride = 1,
                        conv_bias = conv_bias,
                        norm_op = norm_op,
                        norm_op_kwargs = norm_op_kwargs,
                        nonlin = nonlin,
                        nonlin_kwargs = nonlin_kwargs,
                    ) for _ in range(n_blocks_per_stage[s] - 1)
                ]
            )

            mamba_layers.append(
                MambaLayer(
                    dim = np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                    channel_token = do_channel_token[s]
                )
            )

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            # x = self.mamba_layers[s](x)
            x = x + self.mamba_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]


class AttentionUMambaUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        base_channels: int = 16,
        deep_supervision: bool = False,
        multires_input: bool = False,
        act: Union[str, tuple] = ('leakyrelu', {'inplace': True}),
        downsample_mode: str = 'pixel_unshuffle',
        upsample_mode: str = 'pixel_shuffle',
        last_pixelshuffle: bool = False,

        # UMamba specific parameters
        input_size: Tuple[int, ...] = (96, 96, 96),
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.deep_supervision = deep_supervision
        self.multires_input = multires_input
        self.input_size = input_size
        self.use_spectral_norm = use_spectral_norm
        
        c = base_channels
        conv_mod = nn.Conv2d if spatial_dims == 2 else nn.Conv3d

        self.adapt_multires_x2 = conv_mod(32 + c, 32, kernel_size=1)
        self.adapt_multires_x3 = conv_mod(64 + c, 64, kernel_size=1)
        self.adapt_multires_x4 = conv_mod(128 + c, 128, kernel_size=1)
        
        # Use only 3 stages with correct channel progression
        n_stages = 3
        features_per_stage = [2*c, 4*c, 8*c]  # [32, 64, 128] when c=16
        if spatial_dims == 3:
            strides = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
            kernel_sizes = [(3, 3, 3)] * n_stages
        else:
            strides = [(2, 2), (2, 2), (2, 2)]
            kernel_sizes = [(3, 3)] * n_stages
        n_conv_per_stage = [2] * n_stages
        
        for s in range(math.ceil(n_stages / 2), n_stages):
            n_conv_per_stage[s] = 1    
        
        print(f"Deep supervision: {deep_supervision}, Multires input: {multires_input}")
        print(f"Input size: {input_size}, Features per stage: {features_per_stage}")
        print(f"n_conv_per_stage after reduction: {n_conv_per_stage}")
        
        # Original feature extraction layers
        if use_spectral_norm:
            self.feature_conv1 = nn.utils.spectral_norm(conv_mod(in_channels, c, 3, 1, 1))
        else:
            self.feature_conv1 = conv_mod(in_channels, c, 3, 1, 1)
            
        self.out_conv1 = conv_mod(c, out_channels, 3, 1, 1)
        
        if self.deep_supervision:
            self.out_conv2 = conv_mod(2*c, out_channels, 3, 1, 1)
            self.out_conv3 = conv_mod(4*c, out_channels, 3, 1, 1)
            self.out_conv4 = conv_mod(8*c, out_channels, 3, 1, 1)
        
        inc1 = c
        if downsample_mode == 'pixel_unshuffle':
            inc2 = c * (2 ** spatial_dims)
            inc3 = 2*c * (2 ** spatial_dims)
            inc4 = 4*c * (2 ** spatial_dims)
        else:
            inc2, inc3, inc4 = c, 2*c, 4*c
            
        if upsample_mode == 'pixel_shuffle':
            upc1 = 8*c // (2 ** spatial_dims) + 4*c
            upc2 = 4*c // (2 ** spatial_dims) + 2*c
            upc3 = 2*c // (2 ** spatial_dims) + c
        else:
            upc1 = 8*c + 4*c
            upc2 = 4*c + 2*c
            upc3 = 2*c + c
            if last_pixelshuffle:
                upc3 = 2*c // (2 ** spatial_dims) + inc2
        
        if multires_input:
            inc2 += c
            inc3 += c 
            inc4 += c
            if use_spectral_norm:
                self.feature_conv2 = nn.utils.spectral_norm(conv_mod(in_channels, c, 3, 1, 1))
                self.feature_conv3 = nn.utils.spectral_norm(conv_mod(in_channels, c, 3, 1, 1))
                self.feature_conv4 = nn.utils.spectral_norm(conv_mod(in_channels, c, 3, 1, 1))
            else:
                self.feature_conv2 = conv_mod(in_channels, c, 3, 1, 1)
                self.feature_conv3 = conv_mod(in_channels, c, 3, 1, 1)
                self.feature_conv4 = conv_mod(in_channels, c, 3, 1, 1)

        conv_op = convert_dim_to_conv_op(spatial_dims)
        norm_op = get_matching_instancenorm(conv_op)
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        nonlin = nn.LeakyReLU
        nonlin_kwargs = {'inplace': True}
        
        self.calculate_encoder_input_sizes(input_size, strides)
        
        self.umamba_encoder = ResidualMambaEncoder(
            input_size=input_size,
            input_channels=inc1,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            stem_channels=None
        )
        

        self.bottleneck = nn.Sequential(
        nn.utils.spectral_norm(conv_mod(8*c, 8*c, 3, 1, 1)),
        get_act_layer(act),
        nn.utils.spectral_norm(conv_mod(8*c, 8*c, 3, 1, 1)),
        get_act_layer(act),)
        
        self.up_conv1 = nn.Sequential(
            nn.utils.spectral_norm(conv_mod(upc1, 4*c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(4*c, 4*c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(4*c, 4*c, 3, 1, 1)),
            get_act_layer(act),
        )
        
        self.up_conv2 = nn.Sequential(
            nn.utils.spectral_norm(conv_mod(upc2, 2*c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(2*c, 2*c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(2*c, 2*c, 3, 1, 1)),
            get_act_layer(act),
        )
        
        self.up_conv3 = nn.Sequential(
            nn.utils.spectral_norm(conv_mod(upc3, c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(c, c, 3, 1, 1)),
            get_act_layer(act),
            nn.utils.spectral_norm(conv_mod(c, c, 3, 1, 1)),
            get_act_layer(act),
        )
        
        if downsample_mode == 'pixel_unshuffle':
            self.downsample = [PixelUnshuffle(spatial_dims, 2), PixelUnshuffle(spatial_dims, 2), 
                             PixelUnshuffle(spatial_dims, 2)]
        elif downsample_mode in ('nearest', 'trilinear'):
            self.downsample = [
                lambda x: F.interpolate(x, scale_factor=1/2, mode=downsample_mode), 
                lambda x: F.interpolate(x, scale_factor=1/2, mode=downsample_mode), 
                lambda x: F.interpolate(x, scale_factor=1/2, mode=downsample_mode)
            ]
        elif downsample_mode == 'conv':
            self.downsample = [
                conv_mod(c, c, 3, 2, 1),
                conv_mod(2*c, 2*c, 3, 2, 1),
                conv_mod(4*c, 4*c, 3, 2, 1)
            ]
            
        if upsample_mode == 'pixel_shuffle':
            self.upsample = [PixelShuffle(spatial_dims, 2), PixelShuffle(spatial_dims, 2), 
                           PixelShuffle(spatial_dims, 2)]
        elif upsample_mode in ('nearest', 'trilinear'):
            self.upsample = [
                lambda x: F.interpolate(x, scale_factor=2, mode=upsample_mode), 
                lambda x: F.interpolate(x, scale_factor=2, mode=upsample_mode), 
                lambda x: F.interpolate(x, scale_factor=2, mode=upsample_mode)
            ]
            if last_pixelshuffle:
                self.upsample[-1] = PixelShuffle(spatial_dims, 2)

        self.perception_mode = False

        self.merge1 = CBAM(spatial_dims, upc1, 8)
        self.merge2 = CBAM(spatial_dims, upc2, 4)
        self.merge3 = CBAM(spatial_dims, upc3, 2)

    def calculate_encoder_input_sizes(self, input_size, strides):
        """Calculate feature map sizes for encoder stages"""
        self.encoder_input_sizes = []
        current_size = list(input_size)
        self.encoder_input_sizes.append(current_size.copy())
        
        for stride in strides[1:]:
            current_size = [s // st for s, st in zip(current_size, stride)]
            self.encoder_input_sizes.append(current_size.copy())
        
        print(f"Encoder input sizes: {self.encoder_input_sizes}")

    def forward(self, x: Union[torch.Tensor, dict]):
        return_features = self.perception_mode
        out = {}
        features = {}
        
        if isinstance(x, dict):
            x_dict = x
        else:
            x_dict = {'level_0': x}
            
        if self.spatial_dims == 2:
            x_dict = {key: val.squeeze(1).permute(0,3,1,2) for key,val in x_dict.items()}
            
        x = x_dict['level_0']
        feat1 = self.feature_conv1(x)
        
        if self.multires_input:
            interp_mode = 'bilinear' if self.spatial_dims == 2 else 'trilinear'
            if 'level_1' not in x_dict.keys():
                feat2 = self.feature_conv2(F.interpolate(x, scale_factor=1/2, mode=interp_mode))
            else:
                feat2 = self.feature_conv2(x_dict['level_1'])
            if 'level_2' not in x_dict.keys():
                feat3 = self.feature_conv3(F.interpolate(x, scale_factor=1/4, mode=interp_mode))
            else:
                feat3 = self.feature_conv3(x_dict['level_2'])
            if 'level_3' not in x_dict.keys():
                feat4 = self.feature_conv4(F.interpolate(x, scale_factor=1/8, mode=interp_mode))
            else:
                feat4 = self.feature_conv4(x_dict['level_3'])
        
        umamba_outputs = self.umamba_encoder(feat1)
        
        
        x1 = feat1
        x2 = umamba_outputs[0]
        x3 = umamba_outputs[1]
        x4 = umamba_outputs[2]
        x4 = x4 + self.bottleneck(x4) 
        
        if self.multires_input:
            x2 = torch.cat([feat2, x2], dim=1) if 'feat2' in locals() else x2
            x3 = torch.cat([feat3, x3], dim=1) if 'feat3' in locals() else x3
            x4 = torch.cat([feat4, x4], dim=1) if 'feat4' in locals() else x4
            
            x2 = self.adapt_multires_x2(x2)
            x3 = self.adapt_multires_x3(x3)
            x4 = self.adapt_multires_x4(x4)
        
        if self.deep_supervision:
            out['level_3'] = self.out_conv4(x4)
        if return_features:
            features['level_3'] = x4
            
        x = self.upsample[0](x4)
        x = self.up_conv1(self.merge1(torch.cat([x, x3], dim=1)))
        if self.deep_supervision:
            out['level_2'] = self.out_conv3(x)
        if return_features:
            features['level_2'] = x
            
        x = self.upsample[1](x)
        x = self.up_conv2(self.merge2(torch.cat([x, x2], dim=1)))
        if self.deep_supervision:
            out['level_1'] = self.out_conv2(x)
        if return_features:
            features['level_1'] = x
            
        x = self.upsample[2](x)
        x = self.up_conv3(self.merge3(torch.cat([x, x1], dim=1)))
        out['level_0'] = self.out_conv1(x)
        
        if self.spatial_dims == 2:
            for level in out.keys():
                out[level] = out[level].unsqueeze(1).permute(0,1,3,4,2)
                
        if return_features:
            features['level_0'] = x
            output = {
                f"{name}_{level}": o[level]
                for name, o in zip(['feature', 'out'], [features, out])
                for level in o.keys()
            }
            return output
            
        return out