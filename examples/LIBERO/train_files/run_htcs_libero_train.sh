#!/bin/bash
# HTCS — LIBERO 4-suite joint training launcher.
# Mirrors run_libero_train.sh but routes through the HTCS framework + YAML.
#
# Prereq: run codec preprocessing once per dataset before training, otherwise
# HTCSCodecLoader will fail on the first __getitem__ call:
#   python examples/LIBERO/train_files/codec_preprocess.py
set -e

Framework_name=HTCS
base_vlm=playground/Pretrained_models/Qwen3.5-0.8B
config_yaml=./examples/LIBERO/train_files/starvla_htcs_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./playground/Checkpoints
run_id=$(date +%m%d)_htcs_libero4in1

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir} && cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_HTCS
