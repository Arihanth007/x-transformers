#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J llm
#SBATCH --output=logs/data_plafnet.out
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2900
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1


# pretrain LoRA on ZINC
python chemformer_pretrain_zinc.py \
    --task chemformer_dataloader \
    --project uspto50 \
    --run zinc_pretrain \
    --use_rotary_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --grad_accum 1 \
    --vocab_size 576 \
    --learning_rate 1.0 \
    --lr_scheduler func \
    --dividing_factor 10000 \
    --weight_decay 0.0 \
    --num_epochs 10 \
    --validate_every -1 \
    --validate_for -1 \
    --generate_every 10 \
    --device_ids 0 1 2 3 \
    --mask_prob 0.15 \
    --num_workers 16 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True \
    --train True \
    --log True


# pretrain LoRA
# python chemformer_pretrain_lora.py \
#     --task chemformer_dataloader \
#     --project uspto50 \
#     --run best_model \
#     --use_rotary_emb True \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 64 \
#     --grad_accum 8 \
#     --vocab_size 576 \
#     --learning_rate 0.001 \
#     --lr_scheduler onecycle \
#     --dividing_factor 10000 \
#     --weight_decay 0.0 \
#     --num_epochs 1000 \
#     --validate_every -1 \
#     --validate_for -1 \
#     --generate_every 10 \
#     --device_ids 0 1 2 3 \
#     --mask_prob 0 \
#     --num_workers 16 \
#     --save_dir /scratch/arihanth.srikar \
#     --set_precision True \
#     --finetune True \
#     --train True \
#     --log True
