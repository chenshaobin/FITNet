"""
Implementation of dual-scale Transformer and IA unit
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)   # (batch_size, heads, num_patches + 1(or 1), dim_head)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer encoder

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()      # f()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()     # g()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# interactive attention unit

class IAUnit(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        return sm_tokens, lg_tokens

# Dual-scale Transformer encoder with IA unit

class DualScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head=64,
        dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                IAUnit(sm_dim=sm_dim, lg_dim=lg_dim, depth=cross_attn_depth, heads=cross_attn_heads, dim_head=cross_attn_dim_head, dropout=dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# patch-based image feature to token embedder

class ImageFeatureEmbedder(nn.Module):
    def __init__(
        self,
        *,
        channel,
        dim,
        image_size,
        patch_size,
        dropout=0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image feature dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channel * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return self.dropout(x)

# Dual-scale Transformer Block with IA unit

class DST_block_IA(nn.Module):
    def __init__(
        self,
        *,
        sm_image_size,
        sm_channel,
        lg_image_size,
        lg_channel,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size=12,
        sm_enc_depth=1,
        sm_enc_heads=8,
        sm_enc_mlp_dim=2048,
        sm_enc_dim_head=64,    # default
        lg_patch_size=16,
        lg_enc_depth=4,
        lg_enc_heads=8,
        lg_enc_mlp_dim=2048,
        lg_enc_dim_head=64,   # default
        cross_attn_depth=2,
        cross_attn_heads=8,
        cross_attn_dim_head=64,   # default
        depth=3,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        self.sm_image_embedder = ImageFeatureEmbedder(
            channel= sm_channel, dim = sm_dim, image_size = sm_image_size,
            patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = ImageFeatureEmbedder(
            channel= lg_channel, dim = lg_dim, image_size = lg_image_size,
            patch_size = lg_patch_size, dropout = emb_dropout)

        self.dual_scale_encoder = DualScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

    def forward(self, feature1, feature2):
        sm_tokens = self.sm_image_embedder(feature1)
        lg_tokens = self.lg_image_embedder(feature2)
        sm_tokens, lg_tokens = self.dual_scale_encoder(sm_tokens, lg_tokens)
        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))
        fusion_cls = torch.cat((sm_cls, lg_cls), dim=-1)
        return fusion_cls


if __name__ == '__main__':
    v1 = DST_block_IA(
        sm_image_size=112,
        sm_channel=4,
        lg_image_size=56,
        lg_channel=8,
        num_classes=2,
        depth=4,
        sm_dim=1024,
        sm_patch_size=16,
        sm_enc_depth=5,
        sm_enc_heads=8,
        sm_enc_mlp_dim=2048,
        lg_dim=512,
        lg_patch_size=8,
        lg_enc_depth=5,
        lg_enc_heads=8,
        lg_enc_mlp_dim=1024,
        cross_attn_depth=2,
        cross_attn_heads=8,
        dropout=0.1,
        emb_dropout=0.1
    )
    v2 = DST_block_IA(
        sm_image_size = 28,
        sm_channel=16,
        lg_image_size=14,
        lg_channel=32,
        num_classes=2,
        depth=4,
        sm_dim=256,
        sm_patch_size=4,
        sm_enc_depth=5,
        sm_enc_heads=8,
        sm_enc_mlp_dim=512,
        lg_dim=128,
        lg_patch_size=2,
        lg_enc_depth=5,
        lg_enc_heads=8,
        lg_enc_mlp_dim=256,
        cross_attn_depth=2,
        cross_attn_heads=8,
        dropout=0.1,
        emb_dropout=0.1
    )

    feature1 = torch.randn(1, 4, 112, 112)
    feature2 = torch.randn(1, 8, 56, 56)
    pre1 = v1(feature1, feature2)

    feature3 = torch.randn(1, 16, 28, 28)
    feature4 = torch.randn(1, 32, 14, 14)
    pre2 = v2(feature3, feature4)
