import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from transformer import Transformer
from einops import rearrange
from functools import partial
from timm.models.layers.activations import *
from timm.models.efficientnet_blocks import make_divisible


class CNN3D(nn.Module):
    def __init__(self, num_classes=20):
        super(CNN3D, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv3d(7, 100, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv3d(100, 200, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv3d(200, 400, kernel_size=3, stride=1, padding=0)

        # dropout 层
        self.drop1 = nn.Dropout3d(0.3)
        self.drop2 = nn.Dropout3d(0.3)
        self.drop3 = nn.Dropout3d(0.3)
        self.drop4 = nn.Dropout(0.3)

        # 定义池化层
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc1 = nn.Linear(400 * 3 * 3 * 3, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        # 前向传播过程
        ##print("Input shape:", x.shape)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.drop1(x)
        ##print("Conv1 output shape:", x.shape)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.drop2(x)
        ##print("Conv2 output shape:", x.shape)
        x = self.pool1(x)
        ##print("Pool1 output shape:", x.shape)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.drop3(x)
        ##print("Conv3 output shape:", x.shape)
        x = self.pool2(x)
        ##print("Pool2 output shape:", x.shape)
        x = torch.flatten(x, start_dim=1)
        ## print("Flatten output shape:", x.shape)

        x = self.fc1(x)
        x = self.drop4(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        # x = nn.functional.softmax(x, dim=1)

        return x


class ResidueBlock(nn.Module):
    def __init__(self, in_chans=7, f1=50, f2=100, identity=True):
        super().__init__()
        self.s = 1
        if not identity:
            self.s = 2
        self.cbl1 = nn.Sequential(
            nn.Conv3d(in_chans, f1, kernel_size=1, stride=self.s),
            nn.BatchNorm3d(f1),
            nn.ReLU()
        )

        self.cbl2 = nn.Sequential(
            nn.Conv3d(f1, f1, kernel_size=3, padding=1),
            nn.BatchNorm3d(f1),
            nn.ReLU()
        )

        self.cb1 = nn.Sequential(
            nn.Conv3d(f1, f2, kernel_size=1),
            nn.BatchNorm3d(f2),
        )

        self.cb2 = nn.Sequential(
            nn.Conv3d(in_chans, f2, kernel_size=1, stride=self.s),
            nn.BatchNorm3d(f2),
        )

        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.cb2(x)
        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cb1(x)
        x = x + x1
        x = self.act(x)

        return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        kernel_size(int): kernel_size of 3DConv layers
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, kernel_size=7, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, D, H, W, C) -> (N, C, D, H, W)

        x = input + self.drop_path(x)
        return x


class CPDNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        kernel_size(int): kernel_size of 3DConv layers
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, in_chans=7, f1=50, f2=100, identity=True):
        super().__init__()
        if not identity:
            self.s = 2
            self.groups = 1
        else:
            self.s = 1
            self.groups = f1
        self.cbl1 = nn.Sequential(
            nn.Conv3d(in_chans, f1, kernel_size=1, stride=self.s),
            nn.BatchNorm3d(f1),
            nn.ReLU()
        )

        self.cbl2 = nn.Sequential(
            nn.Conv3d(f1, f1, kernel_size=3, padding=1, groups=self.groups),
            nn.BatchNorm3d(f1),
            nn.ReLU()
        )

        self.cb1 = nn.Sequential(
            nn.Conv3d(f1, f2, kernel_size=1),
            nn.BatchNorm3d(f2),
        )

        self.cb2 = nn.Sequential(
            nn.Conv3d(in_chans, f2, kernel_size=1, stride=self.s),
            nn.BatchNorm3d(f2),
        )

        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.cb2(x)
        x = self.cbl1(x)
        x = self.cbl2(x)
        x = self.cb1(x)
        x = x + x1
        x = self.act(x)

        return x


class VitBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw, self.pd = patch_size

        self.conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channel, dim, 1, bias=False),
            nn.BatchNorm3d(dim),
        )

        self.transformer = Transformer(dim, depth, 4, dim, mlp_dim, dropout)

        self.conv3 = nn.Sequential(
            nn.Conv3d(dim, channel, 1, bias=False),
            nn.BatchNorm3d(channel),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(channel + dim, channel, 1, bias=False),
            nn.BatchNorm3d(channel),
        )
        self.act0 = nn.ReLU()
        self.act1 = nn.ReLU()

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        y1 = x.clone()
        x = self.act0(x)
        # Global representations
        _, _, h, w, d = x.shape
        x = rearrange(x, 'b c (h ph) (w pw) (d pd) -> b (ph pw pd) (h w d) c', ph=self.ph, pw=self.pw, pd=self.pd)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw pd) (h w d) c -> b c (h ph) (w pw) (d pd)', h=h // self.ph, w=w // self.pw,
                      d=d // self.pd, ph=self.ph, pw=self.pw, pd=self.pd)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y1), 1)
        x = self.conv4(x)
        x = x + y
        x = self.act1(x)

        return x


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm, eps=eps),
        'ln_3d': partial(LayerNorm3d, eps=eps),
    }
    return norm_dict[norm_layer]


def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'sigmoid': Sigmoid,
        'swish': Swish,
        'mish': Mish,
        'hsigmoid': HardSigmoid,
        'hswish': HardSwish,
        'hmish': HardMish,
        'tanh': Tanh,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'prelu': PReLU,
        'gelu': GELU,
        'silu': nn.SiLU
    }
    return act_dict[act_layer]


class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_3d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv3d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.conv_reduce = nn.Conv3d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv3d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class iRMB(nn.Module):

    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_3d',
                 act_layer='relu', v_proj=True, ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64,
                 window_size=7, attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0.,
                 v_group=False, attn_pre=False, inplace=True):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none',
                                  act_layer='none')
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
                                 norm_layer='none', act_layer=act_layer, inplace=inplace)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
                                     act_layer=act_layer, inplace=inplace)
            else:
                self.v = nn.Identity()
        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=ks, stride=stride, dilation=dilation,
                                      groups=1, norm_layer='bn_3d', act_layer='silu', inplace=inplace)
        self.se = SqueezeExcite(dim_mid, rd_ratio=se_ratio,
                                act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W, D = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H, window_size_D = W, H, D
            else:
                window_size_W, window_size_H, window_size_D = self.window_size, self.window_size, self.window_size
            pad_l, pad_t, pad_f = 0, 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            pad_be = (window_size_D - D % window_size_D) % window_size_D
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, pad_f, pad_be, 0, 0,))
            n1, n2, n3 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W, (D + pad_be) // window_size_D
            x = rearrange(x, 'b c (h1 n1) (w1 n2) (d1 n3) -> (b n1 n2 n3) c h1 w1 d1', n1=n1, n2=n2, n3=n3).contiguous()
            # attention
            b, c, h, w, d = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w d -> qk b heads (h w d) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w d -> b heads (h w d) dim_head',
                              heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w d) dim_head -> b (heads dim_head) h w d', heads=self.num_head,
                                  h=h, w=w, d=d).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w d -> b heads (h w d) dim_head',
                              heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w d) dim_head -> b (heads dim_head) h w d', heads=self.num_head,
                                  h=h, w=w, d=d).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2 n3) c h1 w1 d1 -> b c (h1 n1) (w1 n2) (d1 n3)',
                          n1=n1, n2=n2, n3=n3).contiguous()
            if pad_r > 0 or pad_b > 0 or pad_be > 0:
                x = x[:, :, :H, :W, :D].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, depth, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class LayerNorm3d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w d -> b h w d c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w d c -> b c h w d').contiguous()
        return x


class NeXtCPD(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks at each stage. Default: 1
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=7, num_classes=20, depth=2,
                 layer_scale_init_value=1e-6, head_init_scale=1., drop_path_rate=0.3,
                 ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, 50, kernel_size=3, padding=1),
            LayerNorm(50, eps=1e-6, data_format="channels_first")
        )
        cur = 0
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth * 4)]

        self.stage0 = nn.Sequential(
            *[Block(dim=50, kernel_size=3, drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value) for j in range(depth)]
        )
        cur += depth
        self.dowsample0 = nn.Sequential(
            LayerNorm(50, eps=1e-6, data_format="channels_first"),
            nn.Conv3d(50, 100, kernel_size=2, stride=2),
        )
        self.stage1 = nn.Sequential(
            *[Block(dim=100, kernel_size=3, drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value) for j in range(depth)]
        )

        cur += depth

        self.dowsample1 = nn.Sequential(
            LayerNorm(100, eps=1e-6, data_format="channels_first"),
            nn.Conv3d(100, 200, kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            *[Block(dim=200, kernel_size=3, drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value) for j in range(depth)]
        )
        cur += depth

        self.dowsample2 = nn.Sequential(
            LayerNorm(200, eps=1e-6, data_format="channels_first"),
            nn.Conv3d(200, 400, kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            *[Block(dim=400, kernel_size=3, drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value) for j in range(depth)]
        )
        cur += depth

        self.norm = nn.LayerNorm(400, eps=1e-6)  # final norm layer
        self.head = nn.Linear(400, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage0(x)
        x = self.dowsample0(x)
        x = self.stage1(x)
        x = self.dowsample1(x)
        x = self.stage2(x)
        x = self.dowsample2(x)
        x = self.stage3(x)
        x = self.norm(x.mean([-3, -2, -1]))
        x = self.head(x)
        return x


class NeXtCPDv2(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 7
        num_classes (int): Number of classes for classification head. Default: 1000
    """

    def __init__(self, in_chans=7, num_classes=20):
        super().__init__()

        self.identity1 = nn.Sequential(
            nn.BatchNorm3d(in_chans),
            CPDNeXtBlock(in_chans=in_chans, f1=100, f2=100),
            CPDNeXtBlock(in_chans=100, f1=100, f2=100),
            CPDNeXtBlock(in_chans=100, f1=100, f2=100),
        )
        self.conv1 = CPDNeXtBlock(in_chans=100, f1=100, f2=200, identity=False)

        self.identity2 = nn.Sequential(
            CPDNeXtBlock(in_chans=200, f1=200, f2=200),
            CPDNeXtBlock(in_chans=200, f1=200, f2=200),
            nn.Dropout3d()
        )
        self.conv2 = CPDNeXtBlock(in_chans=200, f1=200, f2=400, identity=False)

        self.identity3 = nn.Sequential(
            CPDNeXtBlock(in_chans=400, f1=400, f2=400),
            CPDNeXtBlock(in_chans=400, f1=400, f2=400),
        )
        self.conv3 = CPDNeXtBlock(in_chans=400, f1=400, f2=800, identity=False)

        self.identity4 = nn.Sequential(
            CPDNeXtBlock(in_chans=800, f1=800, f2=800),
            CPDNeXtBlock(in_chans=800, f1=800, f2=800),
        )

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)

        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(800, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.identity1(x)
        x = self.conv1(x)

        x = self.identity2(x)
        x = self.conv2(x)

        x = self.identity3(x)
        x = self.conv3(x)

        x = self.identity4(x)

        x = self.global_max_pool(x)
        x = self.flatten(x)

        x = self.head(x)

        return x


class MutComputeX(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 7
        num_classes (int): Number of classes for classification head. Default: 1000
    """

    def __init__(self, in_chans=7, num_classes=20):
        super().__init__()

        self.identity1 = nn.Sequential(
            nn.BatchNorm3d(in_chans),
            ResidueBlock(in_chans=in_chans, f1=50, f2=50),
            ResidueBlock(in_chans=50, f1=50, f2=50),
            ResidueBlock(in_chans=50, f1=50, f2=50),
        )
        self.conv1 = ResidueBlock(in_chans=50, f1=50, f2=100, identity=False)

        self.identity2 = nn.Sequential(
            ResidueBlock(in_chans=100, f1=100, f2=100),
            ResidueBlock(in_chans=100, f1=100, f2=100),
        )
        self.conv2 = ResidueBlock(in_chans=100, f1=100, f2=200, identity=False)

        self.identity3 = nn.Sequential(
            ResidueBlock(in_chans=200, f1=200, f2=200),
            ResidueBlock(in_chans=200, f1=200, f2=200),
        )
        self.conv3 = ResidueBlock(in_chans=200, f1=200, f2=400, identity=False)

        self.identity4 = nn.Sequential(
            ResidueBlock(in_chans=400, f1=400, f2=400),
            ResidueBlock(in_chans=400, f1=400, f2=400),
        )

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)

        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.identity1(x)
        x = self.conv1(x)

        x = self.identity2(x)
        x = self.conv2(x)

        x = self.identity3(x)
        x = self.conv3(x)

        x = self.identity4(x)

        x = self.global_max_pool(x)
        x = self.flatten(x)

        x = self.head(x)

        return x


class VitCPD(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
    """

    def __init__(self, in_chans=7, num_classes=20, depth=2, drop_path_rate=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.BatchNorm3d(in_chans),
            ResidueBlock(in_chans=in_chans, f1=50, f2=50),
            ResidueBlock(in_chans=50, f1=50, f2=50),
            ResidueBlock(in_chans=50, f1=50, f2=50),
        )
        self.down1 = ResidueBlock(in_chans=50, f1=50, f2=100, identity=False)

        self.vit1 = VitBlock(100, 2, 100, 3, (2, 2, 2), 200, 0.1)
        self.down2 = ResidueBlock(in_chans=100, f1=100, f2=200, identity=False)

        self.vit2 = VitBlock(200, 4, 200, 3, (5, 5, 5), 800, 0.1)
        self.down3 = ResidueBlock(in_chans=200, f1=200, f2=400, identity=False)

        self.vit3 = VitBlock(400, 3, 400, 3, (3, 3, 3), 800, 0.1)

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)

        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)

        x = self.vit1(x)
        x = self.down2(x)

        x = self.vit2(x)
        x = self.down3(x)

        x = self.vit3(x)
        x = self.global_max_pool(x)
        x = self.flatten(x)

        x = self.head(x)

        return x


class VitCPDv2(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
    """

    def __init__(self, in_chans=7, num_classes=20, depth=2, drop_path_rate=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.BatchNorm3d(in_chans),
            ResidueBlock(in_chans=in_chans, f1=50, f2=50),
        )

        self.vit0 = VitBlock(50, 2, 50, 3, (2, 2, 2), 200)
        self.down1 = ResidueBlock(in_chans=50, f1=50, f2=100, identity=False)

        self.vit1 = VitBlock(100, 2, 100, 3, (2, 2, 2), 200, 0.1)
        self.down2 = ResidueBlock(in_chans=100, f1=100, f2=200, identity=False)

        self.vit2 = VitBlock(200, 4, 200, 3, (5, 5, 5), 800, 0.1)
        self.down3 = ResidueBlock(in_chans=200, f1=200, f2=400, identity=False)

        self.vit3 = VitBlock(400, 3, 400, 3, (3, 3, 3), 800, 0.1)

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)

        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.vit0(x)
        x = self.down1(x)

        x = self.vit1(x)
        x = self.down2(x)

        x = self.vit2(x)
        x = self.down3(x)

        x = self.vit3(x)
        x = self.global_max_pool(x)
        x = self.flatten(x)

        x = self.head(x)

        return x


class EMOCPD(nn.Module):
    r"""
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
    """

    def __init__(self, in_chans=7, num_classes=20, drop_path=0.05):
        super().__init__()

        dprs = [x.item() for x in torch.linspace(0, drop_path, 8)]

        # move stem to blk0, first iRMB in blk0 is stem
        self.blk0 = nn.Sequential(
            iRMB(in_chans, 48, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True,
                 se_ratio=1, exp_ratio=1, dim_head=24, v_proj=False, attn_s=False, window_size=5),
            iRMB(48, 48, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[0],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
            iRMB(48, 48, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[1],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
        )
        self.down0 = iRMB(48, 72, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True, stride=2,
                          se_ratio=0, exp_ratio=2, dim_head=24, v_proj=True, attn_s=False, window_size=5)

        self.blk1 = nn.Sequential(
            iRMB(72, 72, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[2],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
            iRMB(72, 72, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True,  drop_path=dprs[3],
                 se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5),
        )
        self.down1 = iRMB(72, 160, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True, stride=2,
                          se_ratio=0, exp_ratio=1, dim_head=24, v_proj=True, attn_s=False, window_size=5)

        self.blk2 = nn.Sequential(
            iRMB(160, 160, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[4],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=True, window_size=5),
            iRMB(160, 160, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[5],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=True, window_size=5),
        )
        self.down2 = iRMB(160, 288, norm_in=True, has_skip=False, qkv_bias=True, attn_pre=True, stride=2,
                          se_ratio=0, exp_ratio=2, dim_head=32, v_proj=True, attn_s=False, window_size=5)

        self.blk3 = nn.Sequential(
            iRMB(288, 288, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[6],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=1, dim_head=32, v_proj=True, attn_s=True, window_size=3),
            iRMB(288, 288, norm_in=True, has_skip=True, qkv_bias=True, attn_pre=True, drop_path=dprs[7],
                 norm_layer='ln_3d', se_ratio=0, exp_ratio=1, dim_head=32, v_proj=True, attn_s=True, window_size=3),
        )

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)

        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
            nn.Linear(288, 720),
            nn.ReLU(),
            nn.Linear(720, num_classes)
        )

    def forward(self, x):
        x = self.blk0(x)
        x = self.down0(x)

        x = self.blk1(x)
        x = self.down1(x)

        x = self.blk2(x)
        x = self.down2(x)

        x = self.blk3(x)
        x = self.global_max_pool(x)
        x = self.flatten(x)

        x = self.head(x)

        return x
