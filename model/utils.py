import math
import torch


def compute_mask(dims, window_size, shift_size, device):
	"""Computing region masks based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer

	 Args:
		dims: dimension values.
		window_size: local window size.
		shift_size: shift size.
		device: device.
	"""

	cnt = 0

	d, h, w, t = dims
	img_mask = torch.zeros((1, d, h, w, t, 1), device=device)
	for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
		for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
			for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
				for t in slice(-window_size[3]), slice(-window_size[3], -shift_size[3]), slice(-shift_size[3], None):
					img_mask[:, d, h, w, t, :] = cnt
					cnt += 1

	mask_windows = window_partition(img_mask, window_size)
	mask_windows = mask_windows.squeeze(-1)
	attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
	attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

	return attn_mask


def window_partition(x, window_size):
	"""window partition operation based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer

	Partition tokens into their respective windows

	 Args:
		x: input tensor (B, D, H, W, T, C)

		window_size: local window size.


	Returns:
		windows: (B*num_windows, window_size*window_size*window_size*window_size, C)
	"""
	x_shape = x.size()

	b, d, h, w, t, c = x_shape
	x = x.view(
		b,
		d // window_size[0],  # number of windows in depth dimension
		window_size[0],  # window size in depth dimension
		h // window_size[1],  # number of windows in height dimension
		window_size[1],  # window size in height dimension
		w // window_size[2],  # number of windows in width dimension
		window_size[2],  # window size in width dimension
		t // window_size[3],  # number of windows in time dimension
		window_size[3],  # window size in time dimension
		c,
	)
	windows = (
		x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9)
		.contiguous()
		.view(-1, window_size[0] * window_size[1] * window_size[2] * window_size[3], c)
	)
	return windows


def window_reverse(windows, window_size, dims):
	"""window reverse operation based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer

	 Args:
		windows: windows tensor (B*num_windows, window_size, window_size, C)
		window_size: local window size.
		dims: dimension values.

	Returns:
		x: (B, D, H, W, T, C)
	"""

	b, d, h, w, t = dims
	x = windows.view(
		b,
		torch.div(d, window_size[0], rounding_mode="floor"),
		torch.div(h, window_size[1], rounding_mode="floor"),
		torch.div(w, window_size[2], rounding_mode="floor"),
		torch.div(t, window_size[3], rounding_mode="floor"),
		window_size[0],
		window_size[1],
		window_size[2],
		window_size[3],
		-1,
	)
	x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous().view(b, d, h, w, t, -1)

	return x


def get_window_size(x_size, window_size, shift_size=None):
	"""Computing window size based on: "Liu et al.,
	Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
	<https://arxiv.org/abs/2103.14030>"
	https://github.com/microsoft/Swin-Transformer

	 Args:
		x_size: input size.
		window_size: local window size.
		shift_size: window shifting size.
	"""

	use_window_size = list(window_size)
	if shift_size is not None:
		use_shift_size = list(shift_size)
	for i in range(len(x_size)):
		if x_size[i] <= window_size[i]:
			use_window_size[i] = x_size[i]
			if shift_size is not None:
				use_shift_size[i] = 0

	if shift_size is None:
		return tuple(use_window_size)
	else:
		return tuple(use_window_size), tuple(use_shift_size)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
	"""Tensor initialization with truncated normal distribution.
	Based on:
	https://github.com/rwightman/pytorch-image-models

	Args:
	   tensor: an n-dimensional `torch.Tensor`
	   mean: the mean of the normal distribution
	   std: the standard deviation of the normal distribution
	   a: the minimum cutoff value
	   b: the maximum cutoff value
	"""

	if std <= 0:
		raise ValueError("the standard deviation should be greater than zero.")

	if a >= b:
		raise ValueError("minimum cutoff value (a) should be smaller than maximum cutoff value (b).")

	return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
	"""Tensor initialization with truncated normal distribution.
	Based on:
	https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	https://github.com/rwightman/pytorch-image-models

	Args:
	   tensor: an n-dimensional `torch.Tensor`.
	   mean: the mean of the normal distribution.
	   std: the standard deviation of the normal distribution.
	   a: the minimum cutoff value.
	   b: the maximum cutoff value.
	"""

	def norm_cdf(x):
		return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

	with torch.no_grad():
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)
		tensor.uniform_(2 * l - 1, 2 * u - 1)
		tensor.erfinv_()
		tensor.mul_(std * math.sqrt(2.0))
		tensor.add_(mean)
		tensor.clamp_(min=a, max=b)
		return tensor
