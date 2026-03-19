#!/bin/bash

accelerate launch \
  --num_processes 1 \
  --use_deepspeed \
  --deepspeed_config_file deepspeed_zero2_offload.json \
  run_training_ctf_ds.py
