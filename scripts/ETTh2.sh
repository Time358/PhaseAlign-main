#!/bin/bash
# ETTh2: pred_len 96 / 192 / 336 / 720 (hourly, seasonal period 24 = daily cycle)
export CUDA_VISIBLE_DEVICES=0

COMMON="--task_name long_term_forecast --is_training 1 \
  --model_id PhaseAlignDemo --model_comment PhaseAlignDemo --model PhaseAlign \
  --data ETTh2 --root_path ./dataset/ETT-small --data_path ETTh2.csv \
  --seq_len 96 --label_len 48 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 384 --n_heads 8 --e_layers 4 --d_ff 768 --dropout 0.2 \
  --batch_size 64 --eval_batch_size 64 --train_epochs 15 --learning_rate 5e-4 \
  --phase_model PhaseAlign --phase_dim 16 \
  --num_workers 0 --lradj constant \
  --loss MIXED --mse_weight 0.05 --mae_weight 0.95 \
  --use_seasonal_residual 1 --seasonal_period 24 --seasonal_alpha 0.3 --early_stop 0 \
  --seasonal_weights 0.5,0.25,0.15,0.10"

for PL in 96 192 336 720; do
  echo "========== ETTh2 pl=${PL} =========="
  python run_main.py $COMMON --pred_len $PL
done
