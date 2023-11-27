#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J concat_data
#SBATCH --output=logs/data.out
#SBATCH --nodes=1
#SBATCH -c 30
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00

mkdir -p /scratch/arihanth.srikar/
cp data/zinc/zinc-selected.zip /scratch/arihanth.srikar/
cp data/zinc/vocab.txt /scratch/arihanth.srikar/
# rsync -av --progress arihanth.srikar@ada.iiit.ac.in:/share1/arihanth.srikar/zinc-selected.zip /scratch/arihanth.srikar/
export PYTHONUNBUFFERED=1

# absolute
python chemformer.py \
    --task chemformer_dataloader \
    --project uspto50 \
    --run chemloader_absolute \
    --use_pos_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --grad_accum 4 \
    --vocab_size 530 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --validate_every -1 \
    --validate_for -1 \
    --device_ids 2 \
    --mask_prob 0 \
    --num_workers 16 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True \
    --train True \
    --log True

# T5
python chemformer.py \
    --task chemformer_dataloader \
    --project uspto50 \
    --run chemloader_T5 \
    --use_rel_pos_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --grad_accum 4 \
    --vocab_size 530 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --validate_every -1 \
    --validate_for -1 \
    --device_ids 3 \
    --mask_prob 0 \
    --num_workers 16 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True \
    --train True \
    --log True

# T5 eval
python chemformer.py \
    --task chemformer_dataloader \
    --project uspto50 \
    --run chemloader_T5 \
    --use_rel_pos_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --grad_accum 4 \
    --vocab_size 530 \
    --learning_rate 0.001 \
    --num_epochs 100 \
    --validate_every -1 \
    --validate_for -1 \
    --device_ids 3 \
    --mask_prob 0 \
    --num_workers 16 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True

# rotary
python chemformer.py \
    --task chemformer_dataloader \
    --project uspto50 \
    --run onecyle_1000_orig \
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
    --device_ids 0 1 2 3 \
    --mask_prob 0 \
    --num_workers 16 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True \
    --train True \
    --log True

# rotary eval
python chemformer.py \
    --task chemformer_dataloader \
    --project uspto50 \
    --run chemloader_rotary \
    --use_rotary_emb True \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --block_size 512 \
    --batch_size 128 \
    --grad_accum 4 \
    --vocab_size 576 \
    --learning_rate 0.001 \
    --num_epochs 1000 \
    --validate_every -1 \
    --validate_for -1 \
    --generate_every 10 \
    --device_ids 0 \
    --mask_prob 0 \
    --num_workers 16 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True


# pretrain LoRA
python chemformer_pretrain_lora.py \
    --task chemformer_dataloader \
    --project uspto50 \
    --run pretrain_lora_cross_attn \
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
    --device_ids 0 1 2 3 \
    --mask_prob 0 \
    --num_workers 16 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True \
    --train True \
    --log True

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
    --device_ids 0 1 2 3 \
    --mask_prob 0 \
    --num_workers 16 \
    --save_dir /scratch/arihanth.srikar \
    --set_precision True \
    --finetune True \
    --train True \
    --log True
