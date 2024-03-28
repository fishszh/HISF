import torch
import torch.nn as nn
from typing import Union, Tuple
from timm.models.layers import trunc_normal_, DropPath
from functools import partial

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
# from src.networks.UXNet_3D.uxnet_encoder import uxnet_conv

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm3d(dim_in),
                nn.ReLU(),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return nn.functional.normalize(self.proj(x), p=2, dim=1)
    

class UXNET(nn.Module):
    def __init__(
            self,
            in_chans = 1,
            out_chans = 1,
            depths = [2, 2, 2, 2],
            feat_size = [48, 96, 192, 384],
            drop_path_rate = 0,
            layer_scale_init_value = 1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims = 3,
    )-> None:
        """
        Args:
            in_chans: dimension of input channels.
            out_chans: dimension of output channels.
            feat_size: dimension of network feature size.
            hidden_size: dimesion of hidden layer.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if reisdual block is used.
            spatial_dims: number of spatial dims. 
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        self.uxnet_3d = uxnet_conv(
            in_chans = self.in_chans,
            depths = self.depths,
            dims = self.feat_size,
            drop_path_rate = self.drop_path_rate,
            layer_scale_init_value = 1e-6,
            out_indices = self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims = spatial_dims,
            in_channels = self.in_chans,
            out_channels = self.feat_size[0],
            kernel_size = 3,
            stride = 1,
            norm_name = norm_name,
            res_block = res_block
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.feat_size[2], 
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims, 
            in_channels=self.feat_size[3], 
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=[2,2,2],
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        outs = self.uxnet_3d(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        
        # feat = self.conv_proj(dec4)
        
        return self.out(out)
    

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channel_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channel_last', 'channel_first']:
            raise NotImplementedError
        
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channel_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channel_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

        return x
            

class ux_block(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)

        return x


class uxnet_conv(nn.Module):
    """
    Args:
        in_chans (int): Number of input imagae channels. Default: 3
        num_classes (int): Number of classes for classification head. Default 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1

    """
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
            LayerNorm(dims[0], eps=1e-6, data_format="channel_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channel_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ux_block(dim=dims[i], drop_path=dp_rates[cur + j],
                           layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channel_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x




if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    from torchkeras import summary
    import torch
    net = UXNET(
        in_chans=1,
        out_chans=1,
        depths=[2,2,2,2],
        feat_size= [16, 32, 64, 128], #[48,96,192,384],
        drop_path_rate=0,
        layer_scale_init_value=0.,
        hidden_size=768,
        spatial_dims=3
    )
    summary(net, torch.randn((2,1,1024,32,256)))