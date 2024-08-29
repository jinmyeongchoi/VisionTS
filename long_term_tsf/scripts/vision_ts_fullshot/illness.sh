export CUDA_VISIBLE_DEVICES=3
export CKPT_DIR="../ckpt/"
export VM_ARCH="mae_base"
export CONTEXT_LEN=104
export PERIODICITY=52
export ALIGN_CONST=0.4
export NORM_CONST=1.0

for PRED_LEN in 24 36 48 60; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTS \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv  \
    --save_dir save/illness_$PRED_LEN \
    --model_id VisionTS_illness_$PRED_LEN \
    --data custom \
    --features M \
    --train_epochs 100 \
    --vm_arch $VM_ARCH \
    --vm_ckpt $CKPT_DIR \
    --seq_len $CONTEXT_LEN \
    --periodicity $PERIODICITY \
    --pred_len $PRED_LEN \
    --norm_const $NORM_CONST \
    --align_const $ALIGN_CONST
done;