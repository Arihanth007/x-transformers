#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J train_T5
#SBATCH --output=logs/train-%x.%j.out
#SBATCH --nodes=1
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

srun python main.py \
    --task uspto50 \
    --project uspto50 \
    --run T5 \
    --use_rel_pos_emb True \
    --block_size 512 \
    --batch_size 72 \
    --validate_every -1 \
    --validate_for -1 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --train True \
    --log True