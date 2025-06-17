from torch.utils.data import Dataset
import os
from utils import *

	
class SurrogateDataset(Dataset):
	def __init__(self, data_path, std=False, count=-1):
		self.data_path = data_path
		self.data_3d = sorted(os.listdir(f"{data_path}/3d"))
		self.data_2d = sorted(os.listdir(f"{data_path}/2d"))
		self.wet_dry_mask = sorted(os.listdir(f"{data_path}/mask"))

		if count != -1:
			assert len(self.data_3d) >= count, "No such much data"
			self.data_3d = sorted(os.listdir(f"{data_path}/3d"))[:count]
			self.data_2d = sorted(os.listdir(f"{data_path}/2d"))[:count]
			self.wet_dry_mask = sorted(os.listdir(f"{data_path}/mask"))[:count]

		assert len(self.data_3d) == len(self.data_2d) == len(self.wet_dry_mask), "Number of 3d and 2d data do not match"

		self.std = std
		if self.std:
			self.mean_3d = MEAN_3D.view(3, 1, 1, 1, 1)
			self.std_3d = STD_3D.view(3, 1, 1, 1, 1)

			self.mean_2d = MEAN_2D.view(1, 1, 1, 1)
			self.std_2d = STD_2D.view(1, 1, 1, 1)

	def __len__(self):
		return len(self.data_3d)

	def __getitem__(self, idx):
		y_3d = torch.load(f"{self.data_path}/3d/{self.data_3d[idx]}")
		y_2d = torch.load(f"{self.data_path}/2d/{self.data_2d[idx]}")

		mask_3d = y_3d != 0
		mask_2d = torch.load(f"{self.data_path}/mask/{self.wet_dry_mask[idx]}").bool()

		if self.std:
			y_3d = (y_3d - self.mean_3d.to(y_3d.dtype)) / self.std_3d.to(y_3d.dtype)
			y_2d = (y_2d - self.mean_2d.to(y_3d.dtype)) / self.std_2d.to(y_3d.dtype)

		x_3d = torch.zeros(y_3d.shape, dtype=y_3d.dtype)
		x_3d[..., 0] = y_3d[..., 0]
		x_3d[:, [1, -2], :, :, :] = y_3d[:, [1, -2], :, :, :]
		x_3d[:, :, [1, -2], :, :] = y_3d[:, :, [1, -2], :, :]

		x_2d = torch.zeros(y_2d.shape, dtype=y_2d.dtype)
		x_2d[..., 0] = y_2d[..., 0]
		x_2d[:, [1, -2], :, :] = y_2d[:, [1, -2], :, :]
		x_2d[:, :, [1, -2], :] = y_2d[:, :, [1, -2], :]

		mask_3d = mask_3d & (x_3d == 0)
		mask_2d = mask_2d & (x_2d == 0)

		return x_2d, y_2d, mask_2d, x_3d, y_3d, mask_3d
