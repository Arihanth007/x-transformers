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

python temp.py
