#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J finetune_lora
#SBATCH --output=logs/data.out
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH -w gnode118
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1


# finetune LoRA
python chemformer_finetune_lora.py \
    --task chemformer_dataloader \
    --project uspto50 \
    --run finetune_lora_cross_attn \
    --use_rotary_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --grad_accum 4 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --lr_scheduler onecycle \
    --dividing_factor 10000 \
    --weight_decay 0.0 \
    --num_epochs 1000 \
    --validate_every -1 \
    --validate_for -1 \
    --generate_every 10 \
    --device_ids 0 1 \
    --mask_prob 0 \
    --num_workers 16 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True \
    --train True \
    --log True
