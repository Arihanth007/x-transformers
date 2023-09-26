#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J pretrain-zinc-dec
#SBATCH --output=logs/pretrain-%x.%j.out
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
mkdir -p /scratch/arihanth.srikar/data
rsync -av --progress arihanth.srikar@ada.iiit.ac.in:/home2/arihanth.srikar/research/x-transformers/data/zinc /scratch/arihanth.srikar/data
rsync -av --progress arihanth.srikar@ada.iiit.ac.in:/share1/arihanth.srikar/zinc.pkl /scratch/arihanth.srikar/data/zinc
rsync -av --progress arihanth.srikar@ada.iiit.ac.in:/share1/arihanth.srikar/zinc-selected.zip /scratch/arihanth.srikar/data/zinc
export PYTHONUNBUFFERED=1

# pretrain encoder only on zinc
# python pretrain.py \
#     --task zinc \
#     --project uspto50 \
#     --run pretrain-zinc-enc \
#     --use_rotary_emb True \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 128 \
#     --vocab_size 320 \
#     --grad_accum 5 \
#     --validate_every 5000 \
#     --validate_for 400 \
#     --device_ids 0 1 2 3 \
#     --mask_prob 0.15 \
#     --sub_task enc \
#     --num_epochs 1800 \
#     --save_dir /scratch/arihanth.srikar \
#     --set_precision True \
#     --train True \
#     --log True


# pretrain decoder only on zinc
python pretrain.py \
    --task zinc \
    --project uspto50 \
    --run pretrain-zinc-decode \
    --use_rotary_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --vocab_size 320 \
    --grad_accum 5 \
    --validate_every 5000 \
    --validate_for 400 \
    --device_ids 0 1 2 3 \
    --mask_prob 0.0 \
    --sub_task dec \
    --num_epochs 600 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --train True \
    --log True