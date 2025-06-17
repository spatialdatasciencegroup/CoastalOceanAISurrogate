import itertools
from typing import Sequence, Type, Optional
import torch.nn as nn
from torch.nn import LayerNorm
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .utils import *


class PositionalEmbedding(nn.Module):
	"""
	Absolute positional embedding module
	"""

	def __init__(
			self, dim: int, patch_dim: tuple
	) -> None:
		"""
		Args:
			dim: number of feature channels.
			patch_num: total number of patches per time frame
			time_num: total number of time frames
		"""

		super().__init__()
		self.dim = dim
		self.patch_dim = patch_dim
		d, h, w, t = patch_dim
		self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w, 1))
		self.time_embed = nn.Parameter(torch.zeros(1, dim, 1, 1, 1, t))

		trunc_normal_(self.pos_embed, std=0.02)

		trunc_normal_(self.time_embed, std=0.02)

	def forward(self, x):
		b, c, d, h, w, t = x.shape

		x = x + self.pos_embed
		# only add time_embed up to the time frame of the input in case the input size changes
		x = x + self.time_embed[:, :, :, :, :, :t]
		return x


class PatchEmbed(nn.Module):
	""" 4D Image to Patch Embedding
	"""

	def __init__(
			self,
			img_size,
			patch_size,
			in_chans,
			embed_dim,
			norm_layer=None,
			flatten=True,
	):
		assert len(patch_size) == 4, "you have to give four numbers, each corresponds h, w, d, t"
		assert patch_size[3] == 1, "temporal axis merging is not implemented yet"

		super().__init__()
		self.img_size = img_size
		self.patch_size = patch_size
		self.grid_size = (
			img_size[0] // patch_size[0],
			img_size[1] // patch_size[1],
			img_size[2] // patch_size[2],
		)
		self.embed_dim = embed_dim
		self.num_patches = self.grid_size[0] * self.grid_size[1]
		self.flatten = flatten

		self.fc = nn.Linear(in_features=in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3],
							out_features=embed_dim)

		self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

	def forward(self, x):
		torch.cuda.nvtx.range_push("PatchEmbed")
		B, C, H, W, D, T = x.shape
		assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
		assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
		assert D == self.img_size[2], f"Input image width ({D}) doesn't match model ({self.img_size[2]})."
		x = self.proj(x)
		if self.flatten:
			x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
		x = self.norm(x)
		torch.cuda.nvtx.range_pop()
		return x

	def proj(self, x):
		B, C, H, W, D, T = x.shape
		pH, pW, pD = self.grid_size
		sH, sW, sD, sT = self.patch_size

		x = x.view(B, C, pH, sH, pW, sW, pD, sD, -1, sT)
		x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(-1, sH * sW * sD * sT * C)

		x = self.fc(x)

		x = x.view(B, pH, pW, pD, -1, self.embed_dim).contiguous()
		x = x.permute(0, 5, 1, 2, 3, 4)
		return x


class PatchEmbedSurface(nn.Module):
	""" 4D Image to Patch Embedding
	"""

	def __init__(
			self,
			img_size,
			patch_size,
			in_chans,
			embed_dim,
			norm_layer=None,
			flatten=True,
	):
		assert len(patch_size) == 3, "you have to give four numbers, each corresponds h, w, d, t"
		assert patch_size[2] == 1, "temporal axis merging is not implemented yet"

		super().__init__()
		self.img_size = img_size
		self.patch_size = patch_size
		self.grid_size = (
			img_size[0] // patch_size[0],
			img_size[1] // patch_size[1],
		)
		self.embed_dim = embed_dim
		self.num_patches = self.grid_size[0] * self.grid_size[1]
		self.flatten = flatten

		self.fc = nn.Linear(in_features=in_chans * patch_size[0] * patch_size[1] * patch_size[2],
							out_features=embed_dim)

		self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

	def forward(self, x):
		torch.cuda.nvtx.range_push("PatchEmbedSurface")
		B, C, H, W, T = x.shape
		assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
		assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
		x = self.proj(x)
		if self.flatten:
			x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
		x = self.norm(x)
		torch.cuda.nvtx.range_pop()
		return x

	def proj(self, x):
		B, C, H, W, T = x.shape
		pH, pW = self.grid_size
		sH, sW, sT = self.patch_size

		x = x.view(B, C, pH, sH, pW, sW, -1, sT)
		x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(-1, sH * sW * sT * C)
		x = self.fc(x)
		x = x.view(B, pH, pW, -1, self.embed_dim).contiguous()
		x = x.permute(0, 4, 1, 2, 3)
		return x


class PatchMerging(nn.Module):
	"""
	Patch merging layer based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(
			self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, c_multiplier: int = 2
	) -> None:
		"""
		Args:
			dim: number of feature channels.
			norm_layer: normalization layer.
		"""

		super().__init__()
		self.dim = dim

		# Skip dimension reduction on the temporal dimension

		self.reduction = nn.Linear(8 * dim, c_multiplier * dim, bias=False)
		self.norm = norm_layer(8 * dim)

	def forward(self, x):
		x = rearrange(x, "b c d h w t -> b d h w t c")
		x = torch.cat(
			[x[:, i::2, j::2, k::2, :, :] for i, j, k in itertools.product(range(2), range(2), range(2))],
			-1,
		)

		x = self.norm(x)
		x = self.reduction(x)
		x = rearrange(x, "b d h w t c -> b c d h w t")

		return x


class WindowAttention4D(nn.Module):
	"""
	Window based multi-head self attention module with relative position bias based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(
			self,
			dim: int,
			num_heads: int,
			window_size: Sequence[int],
			qkv_bias: bool = False,
			attn_drop: float = 0.0,
			proj_drop: float = 0.0,
	) -> None:
		"""
		Args:
			dim: number of feature channels.
			num_heads: number of attention heads.
			window_size: local window size.
			qkv_bias: add a learnable bias to query, key, value.
			attn_drop: attention dropout rate.
			proj_drop: dropout rate of output.
		"""

		super().__init__()
		self.dim = dim
		self.window_size = window_size
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5
		mesh_args = torch.meshgrid.__kwdefaults__

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x, mask):
		"""Forward function.
		Args:
			x: input features with shape of (num_windows*B, N, C)
			mask: (0/-inf) mask with shape of (num_windows, N, N) or None
		"""
		b_, n, c = x.shape
		qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]
		q = q * self.scale
		attn = q @ k.transpose(-2, -1)
		if mask is not None:
			nw = mask.shape[0]
			attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
			attn = attn.view(-1, self.num_heads, n, n)
			attn = self.softmax(attn)
		else:
			attn = self.softmax(attn)
		attn = self.attn_drop(attn)
		x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class DropPath(nn.Module):
	"""Stochastic drop paths per sample for residual blocks.
	Based on:
	https://github.com/rwightman/pytorch-image-models
	"""

	def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
		"""
		Args:
			drop_prob: drop path probability.
			scale_by_keep: scaling by non-dropped probability.
		"""
		super().__init__()
		self.drop_prob = drop_prob
		self.scale_by_keep = scale_by_keep

		if not (0 <= drop_prob <= 1):
			raise ValueError("Drop path prob should be between 0 and 1.")

	def drop_path(self, x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
		if drop_prob == 0.0 or not training:
			return x
		keep_prob = 1 - drop_prob
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)
		random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
		if keep_prob > 0.0 and scale_by_keep:
			random_tensor.div_(keep_prob)
		return x * random_tensor

	def forward(self, x):
		return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SwinTransformerBlock4D(nn.Module):
	"""
	Swin Transformer block based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(
			self,
			dim: int,
			num_heads: int,
			window_size: Sequence[int],
			shift_size: Sequence[int],
			mlp_ratio: float = 4.0,
			qkv_bias: bool = True,
			drop: float = 0.0,
			attn_drop: float = 0.0,
			drop_path: float = 0.0,
			norm_layer: Type[LayerNorm] = nn.LayerNorm,
			use_checkpoint: bool = False,
	) -> None:
		"""
		Args:
			dim: number of feature channels.
			num_heads: number of attention heads.
			window_size: local window size.
			shift_size: window shift size.
			mlp_ratio: ratio of mlp hidden dim to embedding dim.
			qkv_bias: add a learnable bias to query, key, value.
			drop: dropout rate.
			attn_drop: attention dropout rate.
			drop_path: stochastic depth rate.
			act_layer: activation layer.
			norm_layer: normalization layer.
			use_checkpoint: use gradient checkpointing for reduced memory usage.
		"""

		super().__init__()
		self.dim = dim
		self.num_heads = num_heads
		self.window_size = window_size
		self.shift_size = shift_size
		self.mlp_ratio = mlp_ratio
		self.use_checkpoint = use_checkpoint

		self.norm1 = norm_layer(dim)
		self.attn = WindowAttention4D(
			dim,
			window_size=window_size,
			num_heads=num_heads,
			qkv_bias=qkv_bias,
			attn_drop=attn_drop,
			proj_drop=drop,
		)

		self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim),
								 nn.GELU(),
								 nn.Dropout(drop),
								 nn.Linear(mlp_hidden_dim, dim),
								 nn.Dropout(drop))

	def forward_part1(self, x, mask_matrix):
		b, d, h, w, t, c = x.shape
		window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
		x = self.norm1(x)
		pad_d0 = pad_h0 = pad_w0 = pad_t0 = 0
		pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
		pad_h1 = (window_size[1] - h % window_size[1]) % window_size[1]
		pad_w1 = (window_size[2] - w % window_size[2]) % window_size[2]
		pad_t1 = (window_size[3] - t % window_size[3]) % window_size[3]
		x = F.pad(x, (0, 0, pad_t0, pad_t1, pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1))  # last tuple first in
		_, dp, hp, wp, tp, _ = x.shape
		dims = [b, dp, hp, wp, tp]
		if any(i > 0 for i in shift_size):
			shifted_x = torch.roll(
				x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2], -shift_size[3]), dims=(1, 2, 3, 4)
			)
			attn_mask = mask_matrix
		else:
			shifted_x = x
			attn_mask = None
		x_windows = window_partition(shifted_x, window_size)
		attn_windows = self.attn(x_windows, mask=attn_mask)
		attn_windows = attn_windows.view(-1, *(window_size + (c,)))
		shifted_x = window_reverse(attn_windows, window_size, dims)
		if any(i > 0 for i in shift_size):
			x = torch.roll(
				shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2], shift_size[3]), dims=(1, 2, 3, 4)
			)
		else:
			x = shifted_x

		if pad_d1 > 0 or pad_h1 > 0 or pad_w1 > 0 or pad_t1 > 0:
			x = x[:, :d, :h, :w, :t, :].contiguous()

		return x

	def forward_part2(self, x):
		x = self.drop_path(self.mlp(self.norm2(x)))
		return x

	def forward(self, x, mask_matrix):
		shortcut = x
		if self.use_checkpoint:
			x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
		else:
			x = self.forward_part1(x, mask_matrix)
		x = shortcut + self.drop_path(x)
		# if self.use_checkpoint:
		# 	x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
		# else:
		# 	x = x + self.forward_part2(x)
		x = x + self.forward_part2(x)
		return x


class BasicLayer(nn.Module):
	"""
	Basic Swin Transformer layer in one stage based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(
			self,
			dim: int,
			depth: int,
			num_heads: int,
			window_size: Sequence[int],
			drop_path: list,
			mlp_ratio: float = 4.0,
			qkv_bias: bool = False,
			drop: float = 0.0,
			attn_drop: float = 0.0,
			norm_layer: Type[LayerNorm] = nn.LayerNorm,
			use_checkpoint: bool = False,
	) -> None:
		"""
		Args:
			dim: number of feature channels.
			depth: number of layers in each stage.
			num_heads: number of attention heads.
			window_size: local window size. number of patches
			drop_path: stochastic depth rate.
			mlp_ratio: ratio of mlp hidden dim to embedding dim.
			qkv_bias: add a learnable bias to query, key, value.
			drop: dropout rate.
			attn_drop: attention dropout rate.
			norm_layer: normalization layer.
			downsample: an optional downsampling layer at the end of the layer.
			use_checkpoint: use gradient checkpointing for reduced memory usage.
		"""

		super().__init__()
		self.window_size = window_size
		self.shift_size = tuple(i // 2 for i in window_size)
		self.no_shift = tuple(0 for i in window_size)
		self.depth = depth
		self.use_checkpoint = use_checkpoint
		self.blocks = nn.ModuleList(
			[
				SwinTransformerBlock4D(dim=dim, num_heads=num_heads, window_size=window_size,
									   shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
									   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
									   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
									   norm_layer=norm_layer, use_checkpoint=use_checkpoint)
				for i in range(depth)
			]
		)

	def forward(self, x):
		b, c, d, h, w, t = x.size()
		window_size, shift_size = get_window_size((d, h, w, t), self.window_size, self.shift_size)
		x = rearrange(x, "b c d h w t -> b d h w t c")
		dp = int(np.ceil(d / window_size[0])) * window_size[0]
		hp = int(np.ceil(h / window_size[1])) * window_size[1]
		wp = int(np.ceil(w / window_size[2])) * window_size[2]
		tp = int(np.ceil(t / window_size[3])) * window_size[3]
		attn_mask = compute_mask([dp, hp, wp, tp], window_size, shift_size, x.device)
		for blk in self.blocks:
			x = blk(x, attn_mask)
		x = x.view(b, d, h, w, t, -1)
		x = rearrange(x, "b d h w t c -> b c d h w t")

		return x


class BasicLayerFullAttention(nn.Module):
	"""
	Basic Swin Transformer layer in one stage based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""

	def __init__(
			self,
			dim: int,
			depth: int,
			num_heads: int,
			window_size: Sequence[int],
			drop_path: list,
			mlp_ratio: float = 4.0,
			qkv_bias: bool = False,
			drop: float = 0.0,
			attn_drop: float = 0.0,
			norm_layer: Type[LayerNorm] = nn.LayerNorm,
			use_checkpoint: bool = False,
	) -> None:
		"""
		Args:
			dim: number of feature channels.
			depth: number of layers in each stage.
			num_heads: number of attention heads.
			window_size: local window size.
			drop_path: stochastic depth rate.
			mlp_ratio: ratio of mlp hidden dim to embedding dim.
			qkv_bias: add a learnable bias to query, key, value.
			drop: dropout rate.
			attn_drop: attention dropout rate.
			norm_layer: normalization layer.
			use_checkpoint: use gradient checkpointing for reduced memory usage.
		"""

		super().__init__()
		self.window_size = window_size
		self.shift_size = tuple(i // 2 for i in window_size)
		self.no_shift = tuple(0 for i in window_size)
		self.depth = depth
		self.use_checkpoint = use_checkpoint
		self.blocks = nn.ModuleList(
			[
				SwinTransformerBlock4D(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=self.no_shift,
									   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
									   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
									   norm_layer=norm_layer, use_checkpoint=use_checkpoint)
				for i in range(depth)
			]
		)

	def forward(self, x):
		b, c, d, h, w, t = x.size()
		x = rearrange(x, "b c d h w t -> b d h w t c")
		attn_mask = None
		for blk in self.blocks:
			x = blk(x, attn_mask)
		x = x.view(b, d, h, w, t, -1)
		x = rearrange(x, "b d h w t c -> b c d h w t")

		return x