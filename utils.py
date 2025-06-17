import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset

MEAN_3D = torch.tensor([-3.1893e-03, -2.3433e-03, -9.0075e-07])
STD_3D = torch.tensor([1.2302e-01, 8.8902e-02, 5.7739e-04])

MEAN_2D = torch.tensor([8.9655e-01])
STD_2D = torch.tensor([1.3136e+00])


def sequential_split(dataset, train_ratio):
	train_size = int(len(dataset) * train_ratio)
	indices = list(range(len(dataset)))

	train_indices = indices[:train_size]
	val_indices = indices[train_size:]

	train_subset = Subset(dataset, train_indices)
	val_subset = Subset(dataset, val_indices)

	return train_subset, val_subset


def plot_training_curve(line_1_x, line_1_y, line_2_x, line_2_y, save_path, x_label='Epoch', y_label='Loss'):
	plt.plot(line_1_x, line_1_y, 'o-g', label='Train Dataset', markersize=5)
	plt.plot(line_2_x, line_2_y, 'o-r', label='Valid Dataset', markersize=5)
	plt.xlabel(x_label, size=14)
	plt.ylabel(y_label, size=14)
	plt.title('Training Curve', size=16)
	plt.xticks(size=10)
	plt.yticks(size=10)
	# plt.ylim(0, 0.01)
	plt.legend(loc='lower left', fontsize=15)
	plt.savefig(save_path, dpi=500, bbox_inches="tight")
	plt.clf()


class EarlyStopping:
	"""
	Early stopping to stop the training when the loss does not improve after certain epochs.
	"""

	def __init__(self, patience=5, min_delta=0):
		"""
		:param patience: how many epochs to wait before stopping when loss is not improving
		:param min_delta:
		minimum difference between new loss and old loss for new loss to be considered as an improvement
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False

	def __call__(self, val_loss):
		if self.best_loss == None:
			self.best_loss = val_loss
		elif self.best_loss - val_loss > self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
		elif self.best_loss - val_loss < self.min_delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True