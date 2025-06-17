import logging
import torch.distributed as dist
from tqdm import tqdm
import torch.nn.functional as F

from utils import *


def train(train_loader, val_loader, model, optimizer, scheduler, patience, epoch_num, valid_freq, output_path):
	grad_scaler = torch.cuda.amp.GradScaler()
	early_stop = EarlyStopping(patience)

	train_curve = []
	valid_curve = []
	best = float('inf')

	for epoch in range(epoch_num):
		train_loader.sampler.set_epoch(epoch)

		model.train()
		batch_losses = []

		with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epoch_num}', unit='batch',
		          disable=dist.get_rank() != 0) as pbar:
			for batch in train_loader:
				x_2d, y_2d, mask_2d, x_3d, y_3d, mask_3d = [i.cuda(non_blocking=True) for i in batch]

				with torch.cuda.amp.autocast():
					pred_3d, pred_2d = model((x_3d, x_2d))
					loss = torch.sum(F.mse_loss(pred_3d, y_3d, reduction='none') * mask_3d) / torch.sum(mask_3d) + \
					       torch.sum(F.mse_loss(pred_2d, y_2d, reduction='none') * mask_2d) / torch.sum(mask_2d)

				optimizer.zero_grad()
				grad_scaler.scale(loss).backward()
				grad_scaler.step(optimizer)
				grad_scaler.update()
				pbar.update()

				batch_losses.append(loss.item())

		epoch_loss = torch.tensor(batch_losses).mean().cuda()
		dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
		epoch_loss /= dist.get_world_size()

		scheduler.step(epoch_loss)

		if dist.get_rank() == 0:
			train_curve.append(epoch_loss.item())
			logging.info(f"Epoch {epoch}: Training Loss: {epoch_loss.item()}")
			logging.info(f'Epoch {epoch}: Current learning rate: {scheduler._last_lr}')

		if (epoch + 1) % valid_freq == 0:
			model.eval()
			batch_losses_v = []
			with torch.no_grad():
				with torch.cuda.amp.autocast():
					for batch in val_loader:
						x_2d, y_2d, mask_2d, x_3d, y_3d, mask_3d = [i.cuda(non_blocking=True) for i in batch]
						pred_3d, pred_2d = model((x_3d, x_2d))
						loss = torch.sum(F.mse_loss(pred_3d, y_3d, reduction='none') * mask_3d) / torch.sum(mask_3d) + \
						       torch.sum(F.mse_loss(pred_2d, y_2d, reduction='none') * mask_2d) / torch.sum(mask_2d)

						batch_losses_v.append(loss.item())

			epoch_loss_v = torch.tensor(batch_losses_v).mean().cuda()
			dist.all_reduce(epoch_loss_v, op=dist.ReduceOp.SUM)
			epoch_loss_v /= dist.get_world_size()

			if dist.get_rank() == 0:
				valid_curve.append(epoch_loss_v.item())
				logging.info(f"*******Epoch {epoch}: Validation Loss: {epoch_loss_v.item()}*******")

				if epoch_loss_v < best:
					best = epoch_loss_v
					torch.save(model.module, f"{output_path}/model_best.pth")

			early_stop(epoch_loss_v)
			if early_stop.early_stop:
				logging.info('Early stop!')
				break

		dist.barrier()

	if dist.get_rank() == 0:
		torch.save(model.module, f"{output_path}/model_final.pth")
		plot_training_curve(line_1_x=range(len(train_curve)),
		                    line_1_y=train_curve,
		                    line_2_x=range(valid_freq - 1, valid_freq * len(valid_curve), valid_freq),
		                    line_2_y=valid_curve,
		                    save_path=f"{output_path}/training_curve.png",
		                    y_label="Total MSE")

	dist.destroy_process_group()
