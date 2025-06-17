"""
Our code is based on the following code.
https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html#SwinUNETR
"""

from typing import Sequence, Type, Tuple
import torch.nn as nn
from einops.layers.torch import Rearrange

from .swift_module import PatchEmbed, PositionalEmbedding, PatchMerging, BasicLayerFullAttention, BasicLayer, \
	PatchEmbedSurface
from .utils import *


class Up(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride):
		super().__init__()
		self.up = nn.Sequential(
			nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
			nn.BatchNorm3d(out_channels),
			nn.GELU()
		)
		self.conv = nn.Sequential(
			nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm3d(out_channels),
			nn.GELU(),
			nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm3d(out_channels),
			nn.GELU(),
		)
	
	def forward(self, x_1, x_2):
		x_1 = self.up(x_1)
		x = torch.cat([x_2, x_1], dim=1)
		return self.conv(x)


class SwinUNETR4D(nn.Module):
	"""
	Swin Transformer based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer
	"""
	
	def __init__(
			self,
			img_size_3d: Tuple,
			in_chans_3d: int,
			img_size_2d: Tuple,
			in_chans_2d: int,
			embed_dim: int,
			window_size: Sequence[int],
			first_window_size: Sequence[int],
			patch_size_3d: Sequence[int],
			patch_size_2d: Sequence[int],
			depths: Sequence[int],
			num_heads: Sequence[int],
			mlp_ratio: float = 4.0,
			qkv_bias: bool = True,
			drop_rate: float = 0.0,
			attn_drop_rate: float = 0.0,
			drop_path_rate: float = 0.0,
			norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
			patch_norm: bool = False,
			use_checkpoint: bool = False,
			c_multiplier: int = 2,
			last_layer_full_MSA: bool = False
	) -> None:
		"""
		Args:
			in_chans: dimension of input channels.
			embed_dim: number of linear projection output channels.
			window_size: local window size.
			patch_size: patch size.
			depths: number of layers in each stage.
			num_heads: number of attention heads.
			mlp_ratio: ratio of mlp hidden dim to embedding dim.
			qkv_bias: add a learnable bias to query, key, value.
			drop_rate: dropout rate.
			attn_drop_rate: attention dropout rate.
			drop_path_rate: stochastic depth rate.
			norm_layer: normalization layer.
			patch_norm: add normalization after patch embedding.
			use_checkpoint: use gradient checkpointing for reduced memory usage.
			downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
				user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
				The default is currently `"merging"` (the original version defined in v0.9.0).


			c_multiplier: multiplier for the feature length after patch merging
		"""
		
		super().__init__()
		# encoder
		self.num_layers = len(depths)
		self.patch_embed_3d = PatchEmbed(img_size=img_size_3d, patch_size=patch_size_3d, in_chans=in_chans_3d,
		                                 embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None,
		                                 flatten=False)
		self.patch_embed_2d = PatchEmbedSurface(img_size=img_size_2d, patch_size=patch_size_2d, in_chans=in_chans_2d,
		                                        embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None,
		                                        flatten=False)
		grid_size = list(self.patch_embed_3d.grid_size)
		grid_size[-1] += 1  # add the dim for 2d surface feature
		self.grid_size = grid_size
		self.pos_drop = nn.Dropout(p=drop_rate)
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
		
		patch_dim = ((img_size_3d[0] // patch_size_3d[0]),
		             (img_size_3d[1] // patch_size_3d[1]),
		             (img_size_3d[2] // patch_size_3d[2] + 1),
		             (img_size_3d[3] // patch_size_3d[3]))
		
		# build positional encodings
		self.pos_embeds = nn.ModuleList()
		pos_embed_dim = embed_dim
		for i in range(self.num_layers):
			self.pos_embeds.append(PositionalEmbedding(pos_embed_dim, patch_dim))
			pos_embed_dim = pos_embed_dim * c_multiplier
			patch_dim = (patch_dim[0] // 2, patch_dim[1] // 2, patch_dim[2] // 2, patch_dim[3])
		
		# build layer
		self.layers = nn.ModuleList()
		
		layer_0 = BasicLayer(
			dim=int(embed_dim),
			depth=depths[0],
			num_heads=num_heads[0],
			window_size=first_window_size,
			drop_path=dpr[sum(depths[:0]): sum(depths[: 0 + 1])],
			mlp_ratio=mlp_ratio,
			qkv_bias=qkv_bias,
			drop=drop_rate,
			attn_drop=attn_drop_rate,
			norm_layer=norm_layer,
			use_checkpoint=use_checkpoint,
		)
		
		layer_1 = BasicLayer(
			dim=int(embed_dim * c_multiplier),
			depth=depths[1],
			num_heads=num_heads[1],
			window_size=window_size,
			drop_path=dpr[sum(depths[:1]): sum(depths[: 1 + 1])],
			mlp_ratio=mlp_ratio,
			qkv_bias=qkv_bias,
			drop=drop_rate,
			attn_drop=attn_drop_rate,
			norm_layer=norm_layer,
			use_checkpoint=use_checkpoint,
		)
		
		if not last_layer_full_MSA:
			layer_2 = BasicLayer(
				dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
				depth=depths[(self.num_layers - 1)],
				num_heads=num_heads[(self.num_layers - 1)],
				window_size=window_size,
				drop_path=dpr[sum(depths[: (self.num_layers - 1)]): sum(depths[: (self.num_layers - 1) + 1])],
				mlp_ratio=mlp_ratio,
				qkv_bias=qkv_bias,
				drop=drop_rate,
				attn_drop=attn_drop_rate,
				norm_layer=norm_layer,
				use_checkpoint=use_checkpoint,
			)
		else:
			self.last_window_size = (
				self.grid_size[0] // int(2 ** (self.num_layers - 1)),
				self.grid_size[1] // int(2 ** (self.num_layers - 1)),
				self.grid_size[2] // int(2 ** (self.num_layers - 1)),
				window_size[3],
			)
			
			layer_2 = BasicLayerFullAttention(dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
			                                  depth=depths[(self.num_layers - 1)],
			                                  num_heads=num_heads[(self.num_layers - 1)],
			                                  window_size=self.last_window_size,
			                                  drop_path=dpr[sum(depths[: (
					                                  self.num_layers - 1)]): sum(depths[: (self.num_layers - 1) + 1])],
			                                  mlp_ratio=mlp_ratio,
			                                  qkv_bias=qkv_bias,
			                                  drop=drop_rate,
			                                  attn_drop=attn_drop_rate,
			                                  norm_layer=norm_layer,
			                                  use_checkpoint=use_checkpoint)
		self.layers.append(layer_0)
		self.layers.append(layer_1)
		self.layers.append(layer_2)
		
		self.downsamples = nn.ModuleList([PatchMerging(dim=embed_dim * c_multiplier ** i,
		                                               norm_layer=norm_layer,
		                                               c_multiplier=c_multiplier) for i in range(self.num_layers - 1)])
		
		# decoder
		self.transform = Rearrange('b f h w d t -> (b t) f h w d')
		self.up_blocks = nn.Sequential()
		self.up_blocks.add_module("block0", Up(embed_dim * c_multiplier ** (self.num_layers - 1),
		                                       embed_dim * c_multiplier,
		                                       c_multiplier,
		                                       c_multiplier))
		self.up_blocks.add_module("block1", Up(embed_dim * c_multiplier,
		                                       embed_dim,
		                                       c_multiplier,
		                                       c_multiplier))
		
		self.decode_3d = nn.Sequential(nn.ConvTranspose3d(embed_dim,
		                                                  embed_dim // 2,
		                                                  (patch_size_3d[0], patch_size_3d[1], patch_size_3d[2]),
		                                                  (patch_size_3d[0], patch_size_3d[1], patch_size_3d[2])),
		                               nn.BatchNorm3d(embed_dim // 2),
		                               nn.GELU(),
		                               nn.Conv3d(embed_dim // 2, in_chans_3d, kernel_size=1),
		                               Rearrange('(b t) f h w d -> b f h w d t', t=img_size_3d[-1]))
		
		self.decode_2d = nn.Sequential(nn.ConvTranspose2d(embed_dim,
		                                                  embed_dim // 2,
		                                                  (patch_size_2d[0], patch_size_2d[1]),
		                                                  (patch_size_2d[0], patch_size_2d[1])),
		                               nn.BatchNorm2d(embed_dim // 2),
		                               nn.GELU(),
		                               nn.Conv2d(embed_dim // 2, in_chans_2d, kernel_size=1),
		                               Rearrange('(b t) f h w -> b f h w t', t=img_size_3d[-1]))
	
	def forward(self, x):
		x_3d = x[0]
		x_2d = x[1]
		x = torch.concat([self.patch_embed_2d(x_2d).unsqueeze(-2), self.patch_embed_3d(x_3d)], dim=-2)

		x = self.pos_drop(x)  # (b, c, h, w, d, t)
		
		x0 = self.pos_embeds[0](x)
		x0 = self.layers[0](x0.contiguous())
		
		x1 = self.downsamples[0](x0)
		x1 = self.pos_embeds[1](x1)
		x1 = self.layers[1](x1.contiguous())
		
		x2 = self.downsamples[1](x1)
		x2 = self.pos_embeds[2](x2)
		x2 = self.layers[2](x2.contiguous())
		
		x = self.up_blocks[0](self.transform(x2), self.transform(x1))
		x = self.up_blocks[1](x, self.transform(x0))

		x_2d = x[..., 0]
		x_3d = x[..., 1:]
		out_3d = self.decode_3d(x_3d)
		out_2d = self.decode_2d(x_2d)
		return out_3d, out_2d
