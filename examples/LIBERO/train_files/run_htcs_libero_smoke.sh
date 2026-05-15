#!/bin/bash
# HTCS — single-GPU smoke test (≈200 steps).
# Purpose: verify the full forward/backward path runs end-to-end on one card
# before scaling out. Uses the smallest viable batch + short step budget.
#
# Prereq: run codec preprocessing once first, otherwise HTCSCodecLoader will
# fail on the first __getitem__:
#   python examples/LIBERO/train_files/codec_preprocess.py
set -e

Framework_name=HTCS
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_htcs_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=$(date +%m%d_%H%M)_htcs_smoke

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir} && cp $0 ${output_dir}/

# Disable wandb for smoke runs by default; flip to 'online' once stable.
export WANDB_MODE=${WANDB_MODE:-disabled}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps 200 \
  --trainer.num_warmup_steps 20 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.save_interval 200 \
  --trainer.logging_frequency 10 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_HTCS
