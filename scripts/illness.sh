#!/bin/bash
# illness: weekly data, short series — horizons 24 / 36 / 48 / 60 with seq_len 36
export CUDA_VISIBLE_DEVICES=0

COMMON="--task_name long_term_forecast --is_training 1 \
  --model_id PhaseAlignDemo --model_comment PhaseAlignDemo --model PhaseAlign \
  --data illness --root_path ./dataset/illness --data_path national_illness.csv \
  --freq w \
  --seq_len 36 --label_len 18 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 384 --n_heads 8 --e_layers 4 --d_ff 768 --dropout 0.2 \
  --batch_size 64 --eval_batch_size 64 --train_epochs 15 --learning_rate 5e-4 \
  --phase_model PhaseAlign --phase_dim 16 \
  --num_workers 0 --lradj constant \
  --loss MIXED --mse_weight 0.05 --mae_weight 0.95 \
  --use_seasonal_residual 1 --seasonal_period 52 --seasonal_alpha 0.3 --early_stop 0 \
  --seasonal_weights 0.5,0.25,0.15,0.10"

for PL in 24 36 48 60; do
  echo "========== illness pl=${PL} =========="
  python run_main.py $COMMON --pred_len $PL
done
