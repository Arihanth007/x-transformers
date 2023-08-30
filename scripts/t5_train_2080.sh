#!/bin/bash
#SBATCH -J train_T5
#SBATCH --output=logs/train-%x.%j.out
#SBATCH --nodes=1
#SBATCH -c 9
#SBATCH --gres=gpu:2
#SBATCH --constraint=2080ti
#SBATCH --ntasks-per-node=2
#SBATCH --time=4-00:00:00

mkdir -p /ssd_scratch/users/arihanth.srikar/
export PYTHONUNBUFFERED=1

srun python main.py \
    --task uspto_ifn \
    --project uspto50 \
    --run T5 \
    --use_rel_pos_emb True \
    --block_size 512 \
    --batch_size 18 \
    --grad_accum 4 \
    --validate_every -1 \
    --validate_for -1 \
    --save_dir /ssd_scratch/users/arihanth.srikar \
    --train True \
    --log True