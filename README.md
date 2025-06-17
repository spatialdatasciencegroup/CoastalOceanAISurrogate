# [IPDPS '25] Accelerate Coastal Ocean Circulation Model with AI Surrogate

## Get Started
You can launch the training with the following command:
```bash
torchrun \
  --nnodes <NUM_NODES> \
  --nproc_per_node <NUM_GPUS_PER_NODE> \
  --rdzv_id <UNIQUE_JOB_ID> \
  --rdzv_backend c10d \
  --rdzv_endpoint <MASTER_IP>:<MASTER_PORT> \
  main.py \
  --data_path <DATA_PATH> \
  --output_path <OUTPUT_PATH>
```

## Citation
Please cite our paper if you find this code useful for your work:
```
@inproceedings{xu2025accelerate,
  title={Accelerate Coastal Ocean Circulation Model with AI Surrogate}, 
  author={Zelin Xu and Jie Ren and Yupu Zhang and Jose Maria Gonzalez Ondina and Maitane Olabarrieta and Tingsong Xiao and Wenchong He and Zibo Liu and Shigang Chen and Kaleb Smith and Zhe Jiang},
  booktitle={2025 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  year={2025},
}
```
