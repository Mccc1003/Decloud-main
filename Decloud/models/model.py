import torch
import torch.nn as nn
import torch.nn.functional as F

def maybe_clip(x):
    return torch.clamp(x, min=-1., max=1.)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        if self.pos_embed is None or self.pos_embed.shape[-2:] != x.shape[-2:]:
            self.pos_embed = nn.Parameter(torch.zeros(1, x.shape[1], x.shape[2], x.shape[3]), requires_grad=True).to(x.device)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        x = x + self.pos_embed
        return x

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        q = self.to_q(x).view(B, L, self.heads, self.dim_head).transpose(1,2)  # [B, heads, L, dim_head]
        k = self.to_k(x).view(B, L, self.heads, self.dim_head).transpose(1,2)
        v = self.to_v(x).view(B, L, self.heads, self.dim_head).transpose(1,2)

        q = F.elu(q) + 1
        k = F.elu(k) + 1

        kv = torch.einsum('bhld,bhlv->bhdv', k, v)  # [B, heads, dim_head, dim_head]
        k_sum = k.sum(dim=2)  # [B, heads, dim_head]

        z = 1 / (torch.einsum('bhld,bhd->bhl', q, k_sum) + 1e-6)  # [B, heads, L]

        out = torch.einsum('bhld,bhdv,bhl->bhld', q, kv, z)  # [B, heads, L, dim_head]
        out = out.transpose(1,2).contiguous().view(B, L, C)  # [B, L, C]

        out = self.to_out(out)
        return out

class LightweightTransformer(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, embed_dim=64, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.attention = LinearAttention(embed_dim, heads=4)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.project_out = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, input_x):
        B, C, H, W = input_x.shape

        x = self.patch_embed(input_x)  # [B, embed_dim, H_patch, W_patch]

        H_patch, W_patch = x.shape[2], x.shape[3]
        x_flat = x.flatten(2).transpose(1,2)  # [B, L, embed_dim]

        x_norm = self.norm1(x_flat)
        attn_out = self.attention(x_norm)
        x_flat = x_flat + attn_out

        x_norm = self.norm2(x_flat)
        mlp_out = self.mlp(x_norm)
        x_flat = x_flat + mlp_out

        x = x_flat.transpose(1,2).view(B, self.embed_dim, H_patch, W_patch)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        out = self.project_out(x)  # [B, 6, H, W]

        pred_M = torch.sigmoid(out[:, 0:1])
        pred_T = torch.clamp(torch.sigmoid(out[:, 1:2]), min=1e-6)
        pred_A = torch.sigmoid(out[:, 2:3])
        pred_C = torch.tanh(out[:, 3:6])

        pred_M_rgb = pred_M.expand(-1, 3, -1, -1)
        pred_T_rgb = pred_T.expand(-1, 3, -1, -1)
        pred_A_rgb = pred_A.expand(-1, 3, -1, -1)

        eps = 1e-6
        J_pred = ((input_x - pred_C) / (pred_M_rgb + eps) - pred_A_rgb * (1 - pred_T_rgb)) / (pred_T_rgb + eps)
        J_pred = maybe_clip(J_pred)

        return pred_M_rgb, pred_T_rgb, pred_A_rgb, pred_C, J_pred
