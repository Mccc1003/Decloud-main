import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import torch
import numpy as np
import cv2
from phasepack.phasecong import phasecong
import torch
from torch import nn
import math


def tensor_to_numpy(t):
    return t.detach().cpu().numpy()

# class StructureAwareMeasure:
#     def __init__(self, patch_size=3):
#         self.patch_size = patch_size
#         self.R = 255.0

#     def __call__(self, tensor_image):
#         """
#         输入: tensor_image [1, C, H, W]，单张图像
#         输出: Ms_map [1, 1, H, W]，结构感知图
#         """
#         assert tensor_image.ndim == 4 and tensor_image.shape[0] == 1, "只支持 [1, C, H, W] 输入"

#         # 转灰度 + numpy
#         gray_image = tensor_to_numpy(tensor_image.mean(dim=1).squeeze(0))
#         gray_image = (gray_image * 255).astype(np.uint8)

#         # # 相位一致性分析
#         # M, m, ori, ft, PC, EO, T = phasecong(gray_image, norient=4)
#         # pc_0, pc_45, pc_90, pc_135 = PC[0], PC[1], PC[2], PC[3]

#         H, W = gray_image.shape
#         Ms_map = np.zeros_like(gray_image, dtype=np.float32)
#         image = gray_image.astype(np.float32)

#         # 梯度图
#         gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
#         gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
#         grad_mag = np.sqrt(gx ** 2 + gy ** 2)

#         # J = np.array([[pc_0 * np.dot(gx, gx), pc_45 * np.dot(gx, gy)], [pc_135 * np.dot(gy, gx), pc_90 * np.dot(gy, gy)]])
#         J = np.array([[np.dot(gx, gx), np.dot(gx, gy)], [np.dot(gy, gx), np.dot(gy, gy)]])
#         eigvals = np.linalg.eigvalsh(J)
#         AJ = (np.max(eigvals) - np.min(eigvals)) / (np.sum(eigvals) + 1e-8)
#         Ms_map = AJ * grad_mag / self.R

#         # 归一化并返回 torch tensor
#         Ms_map = (Ms_map - Ms_map.min()) / (Ms_map.max() - Ms_map.min() + 1e-8)
#         Ms_map = torch.from_numpy(Ms_map).float().unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
#         return Ms_map.to(tensor_image.device)

class StructureAwareMeasure:
    def __init__(self, patch_size=3):
        self.patch_size = patch_size
        self.R = 255.0

    def __call__(self, tensor_image):
        """
        输入: tensor_image [1, C, H, W]，单张图像
        输出: Ms_map [1, 1, H, W]，结构感知图
        """
        assert tensor_image.ndim == 4 and tensor_image.shape[0] == 1, "只支持 [1, C, H, W] 输入"

        # 转灰度 + numpy
        gray_image = tensor_to_numpy(tensor_image.mean(dim=1).squeeze(0))
        gray_image = (gray_image * 255).astype(np.uint8)
        
        M, m, ori, ft, PC, EO, T = phasecong(gray_image, norient=4)
        PC = np.max(PC, axis=0)
        PC = torch.from_numpy(PC).float().unsqueeze(0).unsqueeze(1)
        return PC.to(tensor_image.device)

class StructureAwareAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super(StructureAwareAttention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.y_conv = nn.Conv2d(1, dim, kernel_size=1)
        self.query = nn.Conv2d(dim, dim, kernel_size=1)
        self.key = nn.Conv2d(dim, dim, kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, x, y):
        x_query_layer = self.query(x)
        x_key_layer = self.key(x)
        mixed_value_layer = self.value(x)
        B, C, H, W = x.shape
        y = self.y_conv(y)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        y_query_layer = self.query(y)
        y_key_layer = self.key(y)

        query_layer = x_query_layer * y_query_layer
        key_layer = x_key_layer * y_key_layer
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        out = torch.matmul(attention_probs, value_layer)
        out = out.permute(0, 2, 1, 3).contiguous()

        return out

class LinearStructureAwareAttention(nn.Module): 
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads

        self.to_q_x = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_k_x = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias=False)

        self.to_q_y = nn.Conv2d(1, dim, 1, bias=False)
        self.to_k_y = nn.Conv2d(1, dim, 1, bias=False)

        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, y):
        B, C, H, W = x.shape

        # x的Q,K,V
        q_x = self.to_q_x(x).view(B, self.heads, self.dim_head, H*W)  # [B, heads, dim_head, L]
        k_x = self.to_k_x(x).view(B, self.heads, self.dim_head, H*W)
        v = self.to_v(x).view(B, self.heads, self.dim_head, H*W)

        # y的Q,K
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        q_y = self.to_q_y(y).view(B, self.heads, self.dim_head, H*W)
        k_y = self.to_k_y(y).view(B, self.heads, self.dim_head, H*W)

        # 融合Q,K
        q = q_x * q_y
        k = k_x * k_y

        # 核映射，保证非负
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # 计算线性注意力
        kv = torch.einsum('bhcl,bhdl->bhcd', k, v)  # [B, heads, dim_head, dim_head]
        k_sum = k.sum(dim=-1)  # [B, heads, dim_head]

        # 修正einsum维度字符串
        z = 1 / (torch.einsum('bhcl,bhc->bhl', q, k_sum) + 1e-6)  # [B, heads, L]

        out = torch.einsum('bhcl,bhcd,bhl->bhcl', q, kv, z)  # [B, heads, dim_head, L]

        out = out.contiguous().view(B, C, H, W)

        return self.to_out(out)