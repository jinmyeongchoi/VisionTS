#!/bin/bash

#SBATCH --job-name=KL001_t075_ETTh1        # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-10:00:00                     # 1 hour timelimit
#SBATCH --mem=10000MB                         # Using 10GB CPU Memory
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor
#SBATCH --output=../output/ETT/ETTh1/output_ETTh1_KL0001_temp001_pred_full.txt

# --output=output/ETT/ETTh1/output_ETTh1_EM0001_temp075_full.txt

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate

conda activate vision
cd long_term_tsf #/scripts/vision_ts_zeroshot


# export CUDA_VISIBLE_DEVICES=3
export CKPT_DIR="../ckpt/"
export VM_ARCH="mae_base"
export CONTEXT_LEN=1152
export PERIODICITY=24
export ALIGN_CONST=0.4
export NORM_CONST=0.4

for PRED_LEN in 96 192 336 720; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTS \
    --root_path /shared/s2/lab01/dataset/time_series/forecasting/base/ETT-small \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_$PRED_LEN \
    --model_id VisionTS_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 1 \
    --vm_arch $VM_ARCH \
    --vm_ckpt $CKPT_DIR \
    --seq_len $CONTEXT_LEN \
    --periodicity $PERIODICITY \
    --pred_len $PRED_LEN \
    --norm_const $NORM_CONST \
    --align_const $ALIGN_CONST \
    --loss "NEW" \
    --distance "KL" \
    --Lambda 0.001 \
    --temperature 0.01 \
    --temp_to "pred" \
    --base "MSE"
done;


# zhongshsh/CLoT-cn