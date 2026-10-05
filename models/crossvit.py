# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Modifed from Timm. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block

_model_urls = {
    'crossvit_15_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_224.pth',
    'crossvit_15_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_224.pth',
    'crossvit_15_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_15_dagger_384.pth',
    'crossvit_18_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_224.pth',
    'crossvit_18_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_224.pth',
    'crossvit_18_dagger_384': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_18_dagger_384.pth',
    'crossvit_9_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_224.pth',
    'crossvit_9_dagger_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_9_dagger_224.pth',
    'crossvit_base_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_base_224.pth',
    'crossvit_small_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_small_224.pth',
    'crossvit_tiny_224': 'https://github.com/IBM/CrossViT/releases/download/weights-0.1/crossvit_tiny_224.pth',
}


class CrossViT4Branch9_224(nn.Module):
    """4分支特征维度差异化的CrossViT模型"""

    def __init__(self,
                 img_size=[240, 224, 192, 160],  # 4分支输入尺寸
                 patch_size=[12, 16, 10, 20],  # 4分支patch尺寸
                 embed_dim=[128, 256, 192, 384],  # 差异化特征维度
                 depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0], [1, 3, 0]],  # 分支深度
                 num_heads=[4, 8, 6, 12],  # 注意力头数（与维度匹配）
                 mlp_ratio=[3, 3, 4, 4],  # MLP扩展比例
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 drop_block_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.num_branches = 4  # 固定为4分支
        self.embed_dim = embed_dim  # 保存各分支维度

        # 1. 初始化4分支的Patch嵌入层
        self.drop_block_rate = drop_block_rate
        self.patch_embed = nn.ModuleList()
        self.num_patches = []  # 记录每个分支的patch数量
        for i in range(self.num_branches):
            # 计算当前分支的patch数量
            h, w = to_2tuple(img_size[i])
            ph, pw = to_2tuple(patch_size[i])
            num_patches = (h // ph) * (w // pw)
            self.num_patches.append(num_patches)

            # 创建Patch嵌入层（简单卷积实现，可替换为T2T）
            self.patch_embed.append(nn.Conv2d(
                in_channels=3,
                out_channels=embed_dim[i],
                kernel_size=(ph, pw),
                stride=(ph, pw)
            ))

        # 2. 初始化4分支的CLS Token和位置嵌入
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim[i]))
            for i in range(self.num_branches)
        ])
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim[i]))
            for i, num_patches in enumerate(self.num_patches)
        ])
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. 初始化分支内注意力块（Block）
        # 计算drop path速率（每分支按深度分配）
        total_depth = sum(sum(d) for d in depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur_dpr = 0

        self.blocks = nn.ModuleList()  # blocks[分支索引][层索引]
        for branch_idx in range(self.num_branches):
            branch_blocks = nn.ModuleList()
            # 按深度配置添加Block
            for stage_depth in depth[branch_idx]:
                for _ in range(stage_depth):
                    branch_blocks.append(Block(
                        dim=embed_dim[branch_idx],
                        num_heads=num_heads[branch_idx],
                        mlp_ratio=mlp_ratio[branch_idx],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur_dpr],
                        norm_layer=norm_layer
                    ))
                    cur_dpr += 1
            self.blocks.append(branch_blocks)

        # 4. 初始化跨分支注意力（融合4分支特征）
        self.cross_attn = CrossAttention(
            dims=embed_dim,  # 传入4分支维度
            num_heads=num_heads,
            norm_layer=norm_layer
        )

        # 5. 分类头（融合4分支CLS特征）
        self.norm = nn.ModuleList([
            norm_layer(embed_dim[i]) for i in range(self.num_branches)
        ])
        # 融合4分支特征后分类（总维度=128+256+192+384=960）
        self.head = nn.Linear(sum(embed_dim), num_classes) if num_classes > 0 else nn.Identity()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化CLS Token和位置嵌入
        for i in range(self.num_branches):
            trunc_normal_(self.cls_tokens[i], std=.02)
            trunc_normal_(self.pos_embeds[i], std=.02)
        # 初始化其他层
        self.apply(self._init_weights_fn)

    def _init_weights_fn(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x: 输入图像 (B, 3, H, W)
        branch_feats = []

        # 1. 各分支独立提取特征
        for i in range(self.num_branches):
            # 调整输入尺寸适配当前分支
            h, w = to_2tuple(self.img_size[i])
            x_resized = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            # Patch嵌入 + CLS Token + 位置嵌入
            x_patch = self.patch_embed[i](x_resized)  # (B, embed_dim[i], H/ph, W/pw)
            x_patch = x_patch.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim[i])
            cls_token = self.cls_tokens[i].expand(x_patch.shape[0], -1, -1)  # (B, 1, embed_dim[i])
            x_feat = torch.cat((cls_token, x_patch), dim=1)  # (B, num_patches+1, embed_dim[i])
            x_feat = x_feat + self.pos_embeds[i]
            x_feat = self.pos_drop(x_feat)

            # 通过分支内注意力块
            for block in self.blocks[i]:
                x_feat = block(x_feat)

            # 归一化并提取CLS特征
            x_feat = self.norm[i](x_feat)
            branch_feats.append(x_feat[:, 0])  # (B, embed_dim[i])

        # 2. 跨分支注意力融合
        fused_feats = self.cross_attn(branch_feats)  # 输出仍是4分支特征，已交互融合

        # 3. 拼接特征并返回
        return torch.cat(fused_feats, dim=1)  # (B, sum(embed_dim))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# 以下是依赖的子模块（Block和CrossAttention）
class Block(nn.Module):
    """分支内的注意力块（适配单分支维度）"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    """4分支间的交叉注意力（实现特征交互）"""

    def __init__(self, dims, num_heads, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_branches = len(dims)
        self.norm = nn.ModuleList([norm_layer(dim) for dim in dims])
        # 每个分支作为查询，其他分支作为键值对
        self.cross_attns = nn.ModuleList()
        for i in range(self.num_branches):
            self.cross_attns.append(Attention(
                dim=dims[i],
                num_heads=num_heads[i],
                qkv_bias=True,
                # 跨分支注意力需要投影到同一维度
                cross_dim=sum(dims[:i] + dims[i + 1:])  # 其他分支总维度
            ))

    def forward(self, feats):
        # feats: 列表，每个元素为(B, dims[i])
        fused_feats = []
        for i in range(self.num_branches):
            # 当前分支特征作为查询
            q = self.norm[i](feats[i])
            # 其他分支特征拼接作为键值
            kv = torch.cat([feats[j] for j in range(self.num_branches) if j != i], dim=1)
            # 交叉注意力融合
            fused = self.cross_attns[i](q, kv, kv)  # (B, dims[i])
            fused_feats.append(fused + feats[i])  # 残差连接
        return fused_feats


# 以下是基础组件（注意力、MLP、DropPath）
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., cross_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if cross_dim is not None:  # 跨分支时单独定义k/v投影
            self.kv = nn.Linear(cross_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv=None, kv2=None):
        B, N, C = x.shape
        if kv is None:  # 分支内自注意力
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:  # 跨分支注意力（q来自当前分支，k/v来自其他分支）
            q = self.qkv(x)[..., :C].reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(kv).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):#图像尺寸，块大小，通道数，嵌入后的特征维度，布尔值（决定多层卷积还是单层卷积）
        super().__init__()
        img_size = to_2tuple(img_size)#转换为二维数组
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:#多层卷积
            if patch_size[0] == 12:#大小为12则执行4——>3——>1，为16执行422
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),#较大卷积核7*7快速下采样，输出通道数为embed_dim // 4
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)#单层卷积：直接使用一个卷积层将图像分割为块并嵌入

    def forward(self, x):
        B, C, H, W = x.shape#批次大小，通道数，图像高度，宽度
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."#确保输入图像的尺寸 (H, W) 与初始化时指定的 self.img_size 一致。如果不一致，抛出错误提示。
        x = self.proj(x).flatten(2).transpose(1, 2)#self.proj(x)：将输入图像通过 self.proj（在 __init__ 中定义的卷积层或卷积序列）进行投影，生成块嵌入
#如果 multi_conv=False，输出形状为 (B, embed_dim, H//patch_size, W//patch_size)
#如果 multi_conv=True，输出形状类似，但通过多层卷积逐步下采样
#.flatten(2)：从第2个维度（索引从0开始）开始展平张量
#输入形状：(B, embed_dim, num_patches_h, num_patches_w)
#输出形状：(B, embed_dim, num_patches_h * num_patches_w)
#.transpose(1, 2)：交换第1和第2个维度
#输入形状：(B, embed_dim, num_patches)（num_patches = num_patches_h * num_patches_w）
#输出形状：(B, num_patches, embed_dim)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):#num_heads: 注意力头的数量，默认是 8
#qkv_bias: 是否在查询、键、值投影中使用偏置项，默认是 False
#qk_scale: 自定义的查询和键的缩放因子，默认是 None
#attn_drop: 注意力权重的 dropout 比例，默认是 0
#proj_drop: 输出投影的 dropout 比例，默认是 0
        super().__init__()
        self.num_heads = num_heads#计算每个注意力头的维度，即把输入维度 dim 均匀分配给每个注意力头
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        #确定缩放因子。如果 qk_scale 提供了，则使用它；否则，使用默认的缩放因子 1 / sqrt(head_dim)，以防止注意力分数过大
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)#定义三个线性层，用于将输入分别投影到查询（query）、键（key）、值（value）空间
        self.attn_drop = nn.Dropout(attn_drop)#定义一个 dropout 层，用于对注意力权重进行 dropout，以增加模型的鲁棒性
        self.proj = nn.Linear(dim, dim)#定义一个线性层，用于将注意力机制的输出投影回原始维度
        self.proj_drop = nn.Dropout(proj_drop)#定义一个 dropout 层，用于对输出进行 dropout

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        #x[:, 0:1, ...]: 从输入中提取第一个位置的特征（假设这里使用第一个位置的输入作为查询）
        #self.wq(...): 将提取的特征通过查询投影层 wq
        #reshape(B, 1, self.num_heads, C // self.num_heads): 将投影后的特征重新形状化为 (B, 1, num_heads, head_dim)，以便于多头注意力计算
        #permute(0, 2, 1, 3): 调整张量的维度顺序，使其变为 (B, num_heads, 1, head_dim)，以便于后续的矩阵乘法
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #k.transpose(-2, -1): 转置键矩阵的倒数第二和倒数第一维度，使其形状变为 (B, num_heads, head_dim, N)。
        #q @ k.transpose(-2, -1): 计算查询和键之间的点积，得到注意力分数矩阵。
        #* self.scale: 使用之前定义的缩放因子对注意力分数进行缩放，防止分数过大。
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):#交叉注意力，与自注意力不同，CrossAttention 的Q来自当前序列，K/V来自其他序列

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):#dim：列表，每个分支的特征维度（如 [64, 128, 256]）。
#patches：块，但是在这里并未被使用。
#depth：列表，每个分支的Block数量（如 [2, 2, 2]）。
#num_heads：列表，每个分支的注意力头数（如 [4, 8, 16]）。
#mlp_ratio：列表，每个分支的MLP隐藏层扩展比例（如 [4., 4., 4.]）。
#其他参数（qkv_bias, drop等）与标准Transformer一致。
        super().__init__()
        last_depth = depth[-1]
        if isinstance(last_depth, list):
            total_last_depth = sum(last_depth)  # 如果是列表，计算总深度
        else:
            total_last_depth = last_depth  # 如果是整数，直接使用
        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()


        for d in range(num_branches):
            tmp = []
            # 处理当前分支的深度（可能是整数或列表）
            current_depth = depth[d]
            # 如果是整数，转换为单元素列表以统一处理逻辑
            if isinstance(current_depth, int):
                current_depth = [current_depth]
            # 遍历当前分支的每个阶段深度
            for stage_depth in current_depth:
                # 为每个阶段添加对应数量的 Block
                for _ in range(stage_depth):
                    # 使用当前已添加的 Block 数量作为 drop_path 索引（避免越界）
                    tmp.append(
                        Block(dim=dim[d],
                              num_heads=num_heads[d],
                              mlp_ratio=mlp_ratio[d],
                              qkv_bias=qkv_bias,
                              drop=drop,
                              attn_drop=attn_drop,
                              drop_path=drop_path[len(tmp)] if len(drop_path) > 0 else 0.,
                              norm_layer=norm_layer)
                    )
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

            # 修复融合层对 depth[-1] 的处理（同样兼容整数和列表）
        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            # 处理最后一个分支的深度（可能是整数或列表）
            last_depth = depth[-1]
            if isinstance(last_depth, list):
                total_fusion_depth = sum(last_depth)  # 列表则求和
            else:
                total_fusion_depth = last_depth  # 整数则直接使用
            # 根据总深度创建融合层
            if total_fusion_depth == 0:
                self.fusion.append(CrossAttentionBlock(
                    dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[-1] if len(drop_path) > 0 else 0.,
                    norm_layer=norm_layer, has_mlp=False
                ))
            else:
                tmp = []
                for _ in range(total_fusion_depth):
                    tmp.append(CrossAttentionBlock(
                        dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d],
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[-1] if len(drop_path) > 0 else 0.,
                        norm_layer=norm_layer, has_mlp=False
                    ))
                self.fusion.append(nn.Sequential(*tmp))
#为每个分支创建一组连续的 Block（可能是Transformer Block）。每个分支的Block数量由 depth[d] 决定。如果所有分支的 depth 为0，则 self.blocks 设为 None。
        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches] and False:#如果维度相同（dim[d] == dim[d+1]），默认使用恒等映射（nn.Identity），但条件 and False 禁用了此逻辑。
                tmp = [nn.Identity()]
            else:#通过 LayerNorm + 激活函数 + 线性层 调整维度
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))
#将当前分支的特征投影到下一个分支的维度（实现跨分支交互）。
        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if total_last_depth == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                       has_mlp=False))#如果 total_last_depth == 0，使用单个 CrossAttentionBlock；否则堆叠多个。
            else:
                tmp = []
                for _ in range(total_last_depth):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))
#对每个分支 d，与下一个分支 d+1 进行交互。has_mlp=False 表示融合时不使用MLP。
        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d+1) % num_branches]), act_layer(), nn.Linear(dim[(d+1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))
#将融合后的特征投影回原始分支的维度
    def forward(self, x):#对cls token进行处理
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]#对每个分支的输入 x_ 应用对应的 block 处理，得到cls token，即outs_b
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]#self.projs 是一组投影层，将每个分支的CLS token投影到另一个空间
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)#交叉拼接
            tmp = self.fusion[i](tmp)#第i个融合模块，处理拼接过的特征
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])#将融合后的cls token还原到原始空间
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)#重新拼接原始特征
            outs.append(tmp)#将处理后的多分支特征列表存入out
        return outs


def _compute_num_patches(img_size, patches):
    return [i // p * i // p for i, p in zip(img_size,patches)]


class VisionTransformer(nn.Module):
    def __init__(self, img_size=(224, 224, 192, 160),  # 新增4个分支的图像尺寸
                 patch_size=(8, 16, 10, 20),  # 4个分支的patch尺寸
                 in_chans=3, num_classes=1000,
                 embed_dim=(192, 384, 128, 256),  # 4个分支的嵌入维度
                 depth=([1,3,1], [1,3,1], [1,3,1], [1,3,1]),  # 4个分支的深度
                 num_heads=(6, 12, 4, 8),  # 4个分支的注意力头数
                 mlp_ratio=(2., 2., 4., 3.),  # 4个分支的MLP比例
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,drop_block_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm,
                 multi_conv=False, pretrained_cfg=None, pretrained_cfg_overlay=None, cache_dir=None):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = [to_2tuple(img_size)] * 4  # 确保4个分支的尺寸列表
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)  # 改为4个分支

        self.patch_embed = nn.ModuleList()
        if hybrid_backbone is None:
            # 初始化4个分支的位置嵌入
            self.pos_embed = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i]))
                for i in range(self.num_branches)
            ])
            # 初始化4个分支的patch嵌入
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(
                    img_size=im_s, patch_size=p, in_chans=in_chans,
                    embed_dim=d, multi_conv=multi_conv
                ))
        else:
            # 适配T2T模块的4分支初始化（保持与原逻辑一致，扩展为4分支）
            self.pos_embed = nn.ParameterList()
            from .t2t import T2T, get_sinusoid_encoding
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(T2T(
                    im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d
                ))
                self.pos_embed.append(nn.Parameter(
                    data=get_sinusoid_encoding(n_position=1 + num_patches[idx], d_hid=embed_dim[idx]),
                    requires_grad=False
                ))

            del self.pos_embed
            self.pos_embed = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i]))
                for i in range(self.num_branches)
            ])

        # 初始化4个分支的CLS token
        self.cls_token = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim[i]))
            for i in range(self.num_branches)
        ])
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 调整深度计算（适配4分支）
        total_depth = 0
        for branch_depth in depth:
            if isinstance(branch_depth, list):
                total_depth += sum(branch_depth)  # 分支深度为列表时，累加所有元素
            else:
                total_depth += branch_depth  # 分支深度为整数时，直接累加

        # 生成drop_path列表（长度与总层数一致）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dpr_ptr = 0  # 指针用于分配每个分支的dpr

        self.blocks = nn.ModuleList()
        for idx, block_cfg in enumerate(depth):
            # 计算当前分支的总层数
            curr_depth = sum(block_cfg) if isinstance(block_cfg, list) else block_cfg
            # 分配当前分支的dpr片段（从dpr_ptr开始，长度为curr_depth）
            dpr_ = dpr[dpr_ptr: dpr_ptr + curr_depth]
            blk = MultiScaleBlock(
                dim=embed_dim,  # 传入4分支维度
                patches=num_patches,
                depth=depth,  # 传入所有分支的深度配置
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_,  # 传入当前分支的dpr片段
                norm_layer=norm_layer
            )
            dpr_ptr += curr_depth  # 移动指针到下一分支
            self.blocks.append(blk)

        # 4个分支的归一化和分类头
        self.norm = nn.ModuleList([
            norm_layer(embed_dim[i]) for i in range(self.num_branches)
        ])
        self.head = nn.ModuleList([
            nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()
            for i in range(self.num_branches)
        ])

        # 权重初始化（保持不变）
        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)
        self.apply(self._init_weights)
#下为其他方法
    def _init_weights(self, m):#权重初始化法
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):#无权重衰减的参数
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, C, H, W = x.shape
        xs = []
        # 处理4个分支的输入
        for i in range(self.num_branches):
            # 调整输入图像尺寸以匹配当前分支
            if H != self.img_size[i][0] or W != self.img_size[i][1]:
                x_ = torch.nn.functional.interpolate(
                    x, size=(self.img_size[i][0], self.img_size[i][1]), mode='bicubic'
                )
            else:
                x_ = x
            # 生成patch嵌入并拼接CLS token
            tmp = self.patch_embed[i](x_)
            cls_tokens = self.cls_token[i].expand(B, -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        # 多分支特征交互
        for blk in self.blocks:
            xs = blk(xs)

        # 归一化并提取CLS token
        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]  # 取每个分支的CLS token
        return out

    def forward(self, x):
        xs = self.forward_features(x)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = torch.mean(torch.stack(ce_logits, dim=0), dim=0)
        return ce_logits
#能够处理不同输入尺寸并通过CLS token聚合全局信息用于分类任务。



@register_model
def crossvit_tiny_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_tiny_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_small_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[6, 6], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_small_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_base_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[384, 768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[12, 12], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_base_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_9_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_9_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_15_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_18_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def crossvit_9_dagger_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_9_dagger_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def crossvit_15_dagger_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_dagger_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def crossvit_15_dagger_384(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_15_dagger_384'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def crossvit_18_dagger_224(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_224'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def crossvit_18_dagger_384(pretrained=False, **kwargs):
    model = VisionTransformer(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_18_dagger_384'], map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def crossvit_4branch_9_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=[240, 224, 192, 160],  # 4个分支的输入图像尺寸（逐步减小）
        patch_size=[12, 16, 10, 20],    # 4个分支的patch尺寸（与尺寸匹配）
        embed_dim=[128, 256, 192, 384], # 4个分支的特征维度（差异化设计）
        depth=[
            [1, 3, 0],  # 分支1的深度配置
            [1, 3, 0],  # 分支2的深度配置
            [1, 3, 0],  # 分支3的深度配置
            [1, 3, 0]   # 分支4的深度配置
        ],
        num_heads=[4, 8, 6, 12],        # 4个分支的注意力头数（与维度匹配）
        mlp_ratio=[3, 3, 4, 4],         # 4个分支的MLP扩展比例（差异化）
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_block_rate=kwargs.get('drop_block_rate', 0.),
        ** kwargs
    )
    model.default_cfg = _cfg()
    # 若有预训练权重，可在此添加加载逻辑
    # if pretrained:
    #     state_dict = torch.hub.load_state_dict_from_url(_model_urls['crossvit_4branch_9_224'], map_location='cpu')
    #     model.load_state_dict(state_dict)
    if pretrained:
        raise ValueError("No pretrained weights for crossvit_4branch_9_224")
    return model
print(Block.__init__)  # 查看方法定义