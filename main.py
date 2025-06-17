import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import time

from train import *
from dataset import *
from model import *


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_path', default=None, type=str)
	parser.add_argument('--data_path', default=os.environ['SLURM_TMPDIR'], type=str)
	parser.add_argument('--data_count', default=-1, type=int)
	parser.add_argument('--std', action='store_true', default=False)

	# training parameters
	parser.add_argument('--epoch_num', default=40, type=int)
	parser.add_argument('--batch_size', default=2, type=int)
	parser.add_argument('--num_workers', default=6, type=int)
	parser.add_argument('--learning_rate', default=1e-3, type=float)
	parser.add_argument('--scheduler_patience', default=3, type=int)
	parser.add_argument('--weight_decay', default=3e-6, type=float)
	parser.add_argument('--early_stop_patience', default=2, type=int)
	parser.add_argument('--valid_freq', default=5, type=int)
	parser.add_argument('--pin_memory', action='store_false', default=True)

	# models parameters
	parser.add_argument('--embed_dim', default=24, type=int)
	parser.add_argument('--predict_steps', default=24, type=int)
	parser.add_argument('--depths', default=(2, 2, 2), type=tuple)
	parser.add_argument('--num_heads', default=(3, 6, 12), type=tuple)
	parser.add_argument('--patch_size_3d', default=(5, 5, 4, 1), type=tuple)
	parser.add_argument('--patch_size_2d', default=(5, 5, 1), type=tuple)
	parser.add_argument('--first_window_size', default=(2, 2, 2, 2), type=tuple)
	parser.add_argument('--window_size', default=(4, 4, 2, 2), type=tuple)
	parser.add_argument('--last_layer_full_MSA', action='store_false', default=True)
	parser.add_argument('--checkpoint', action='store_false', default=True)

	return parser.parse_args()


def main():
	# initialize DDP
	rank = int(os.environ["RANK"])
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend='nccl')

	args = parse_args()

	# initialize log
	start_time = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
	output_root = 'output/'
	if args.output_path is None:
		output_path = output_root + start_time + '/'
	else:
		output_path = output_root + args.output_path + '/'

	if dist.get_rank() == 0:
		if not os.path.exists(output_root):
			os.mkdir(output_root)
		if not os.path.exists(output_path):
			os.mkdir(output_path)
		logging.basicConfig(filename=f"{output_path}/log.log", filemode='a', datefmt='%H:%M:%S',
		                    level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
		                    format='%(asctime)s: %(message)s')

	logging.info(args)

	# dataset
	dataset_all = SurrogateDataset(args.data_path, std=args.std, count=args.data_count)
	train_set, val_set = sequential_split(dataset_all, 0.9)
	train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False)
	val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)

	train_loader = DataLoader(train_set,
	                          batch_size=args.batch_size,
	                          num_workers=args.num_workers,
	                          pin_memory=args.pin_memory,
	                          sampler=train_sampler)
	val_loader = DataLoader(val_set,
	                        batch_size=args.batch_size,
	                        num_workers=args.num_workers,
	                        sampler=val_sampler,
	                        drop_last=True)

	img_size_2d = train_set[0][0].shape[1:]
	in_chans_2d = train_set[0][0].shape[0]
	img_size_3d = train_set[0][3].shape[1:]
	in_chans_3d = train_set[0][3].shape[0]
	logging.info(f'''
	train_size: {len(train_set)}
	val_size: {len(val_set)}
	img_size_2d: {img_size_2d}
	in_chans_2d: {in_chans_2d}
	img_size_3d: {img_size_3d}
	in_chans_3d: {in_chans_3d}
	''')

	# initialize model
	model = SwinUNETR4D(img_size_3d=img_size_3d,
	                    in_chans_3d=in_chans_3d,
	                    img_size_2d=img_size_2d,
	                    in_chans_2d=in_chans_2d,
	                    embed_dim=args.embed_dim,
	                    window_size=args.window_size,
	                    first_window_size=args.first_window_size,
	                    patch_size_3d=args.patch_size_3d,
	                    patch_size_2d=args.patch_size_2d,
	                    depths=args.depths,
	                    num_heads=args.num_heads,
	                    use_checkpoint=args.checkpoint,
	                    last_layer_full_MSA=args.last_layer_full_MSA).cuda()

	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model = DDP(model, device_ids=[local_rank], output_device=local_rank)

	# training set up
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, factor=0.1,
	                                                       threshold=1e-3)

	train(train_loader,
	      val_loader,
	      model,
	      optimizer,
	      scheduler,
	      args.early_stop_patience,
	      args.epoch_num,
	      args.valid_freq,
	      output_path)


if __name__ == "__main__":
	main()
