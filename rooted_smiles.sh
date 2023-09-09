#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J rooted-smiles-3200-vocab
#SBATCH --output=logs/train-%x.%j.out
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

python main.py \
    --task rooted_smiles \
    --project uspto50 \
    --run rooted-smiles-3200-vocab \
    --use_rel_pos_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --vocab_size 3200 \
    --vocab_file 2000 \
    --grad_accum 1 \
    --validate_every -1 \
    --validate_for -1 \
    --device_ids 0 1 2 3 \
    --mask_prob 0.15 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --train True \
    --log True


# python main.py \
#     --task rooted_smiles \
#     --project uspto50 \
#     --run rooted_smiles \
#     --use_rel_pos_emb True \
#     --n_layer 6 \
#     --n_head 8 \
#     --n_embd 512 \
#     --block_size 512 \
#     --batch_size 128 \
#     --vocab_size 256 \
#     --grad_accum 1 \
#     --validate_every -1 \
#     --validate_for -1 \
#     --device_ids 0 1 2 3 \
#     --mask_prob 0.15 \
#     --save_dir /scratch/arihanth.srikar \
#     --set_precision True \
#     --train True \
#     --log True