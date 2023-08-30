#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J zinc
#SBATCH --output=logs/pretrain-%x.%j.out
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
cp -r data /scratch/arihanth.srikar/
rsync -av --progress arihanth.srikar@ada.iiit.ac.in:/share1/arihanth.srikar/zinc.pkl /scratch/arihanth.srikar/data/zinc
rsync -av --progress arihanth.srikar@ada.iiit.ac.in:/share1/arihanth.srikar/zinc-selected.zip /scratch/arihanth.srikar/data/zinc
export PYTHONUNBUFFERED=1

python main.py \
    --task zinc \
    --project uspto50 \
    --run pretrain \
    --use_rel_pos_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 384 \
    --vocab_size 128 \
    --grad_accum 1 \
    --validate_every 1000 \
    --validate_for 200 \
    --device_ids 0 1 2 3 \
    --mask_prob 0.15 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --train True \
    --log True