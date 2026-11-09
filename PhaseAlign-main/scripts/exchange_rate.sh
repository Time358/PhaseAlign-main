#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

COMMON=(--task_name long_term_forecast --is_training 1
        --model_id PhaseAlign_Exchange --model_comment PhaseAlign
        --data exchange --root_path ./dataset/exchange_rate --data_path exchange_rate.csv
        --features M --target OT --freq d
        --seq_len 96 --label_len 48
        --enc_in 8
        --d_model 384 --n_heads 8 --e_layers 4 --d_ff 768 --dropout 0.2
        --patch_len 32 --stride 16 --top_k 9
        --phase_weight 0.1 --align_weight 0.1
        --batch_size 32 --eval_batch_size 8 --train_epochs 10
        --learning_rate 1e-4 --patience 10
        --seeds 2025,2026,2027 --num_workers 0)

for PL in 96 192 336 720; do
  echo "========== Exchange H=${PL} =========="
  python run_main.py "${COMMON[@]}" --pred_len ${PL}
done
