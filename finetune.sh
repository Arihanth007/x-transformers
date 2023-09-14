#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J finetune-rooted
#SBATCH --output=logs/finetune-rooted-%x.%j.out
#SBATCH -c 36
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

# finetune on rooted smiles (pretrained on zinc)
python pretrain.py \
    --task finetune_rootes_smiles \
    --project uspto50 \
    --run finetune_rootes_smiles_dropout \
    --use_rel_pos_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --vocab_size 320 \
    --vocab_file _pretrain \
    --grad_accum 1 \
    --validate_every -1 \
    --validate_for -1 \
    --device_ids 0 1 2 3 \
    --mask_prob 0 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True \
    --train True \
    --log True



# finetune on smiles (pretrained on zinc)
python pretrain.py \
    --task finetune_rootes_smiles \
    --project uspto50 \
    --run finetune_rootes_smiles_dropout \
    --use_rel_pos_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --vocab_size 320 \
    --vocab_file data/rooted/vocab_pretrain.txt \
    --grad_accum 1 \
    --validate_every -1 \
    --validate_for -1 \
    --device_ids 0 1 2 3 \
    --mask_prob 0 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True \
    --train True \
    --log True
