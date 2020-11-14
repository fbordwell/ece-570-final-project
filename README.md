# ECE 570 Final Project

### 1. Running the experiment

Create the conda environment "unetgan" from the provided unetgan.yml file. The experiments can be reproduced with the scripts provided in the folder `training_scripts` (the experiment folder and dataset folder has to be set manually). My exact build configuration for the FFHQ dataset has some modifications and was as follows:

```bash
python3 train.py --dataset FFHQ --parallel --shuffle --which_best FID --batch_size 3 --num_G_accumulations 1 --num_D_accumulations 1 --num_D_steps 1 --G_lr 1e-4 --D_lr 5e-4 --D_B2 0.999 --G_B2 0.999 --G_attn 0 --D_attn 0 --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 --G_ortho 0.0 --seed 99 --G_init ortho --D_init ortho --G_eval_mode --G_ch 64 --D_ch 64 --hier --dim_z 128 --ema --use_ema --ema_start 21000 --accumulate_stats --num_standing_accumulations 100 --test_every 10000 --save_every 10000 --num_best_copies 2 --num_save_copies 1 --seed 0 --sample_every 30 --id ffhq_unet_bce_noatt_cutmix_consist --gpus "0,1" --unconditional --unet_mixup --slow_mixup --full_batch_mixup --consistency_loss_and_augmentation --warmup_epochs 100 --base_root results --data_folder images\ffhq
```

### 2. Files

The following files are from the [U-Net GAN](https://github.com/boschresearch/unetgan) repository and are unmodified:

```
BigGAN.py
calculate_inception_moments.py
datasets.py
inception.py
inception_utils.py
layers.py
losses.py
PyTorchDatasets.py
train_fns.py
```

The following files are from the same repository and are modified to test different build configurations, batch sizes, and network structures:

```
spectral.py
train.py
utils.py
```

The following files are made by me to calculate FID scores:

```
fid_plot.py
fid_score.py
fid_stats.py
```

`move.sh` is a script made by me to move files into the proper folders to replicate the original repository.

### 3. Datasets

I used the FFHQ dataset for testing. Images should be resized to 256x256, so it's simpler to download the scaled dataset [here](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/122786). Files should be partitioned into folders in the method shown [here](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL). `move.sh` can be used for this.
