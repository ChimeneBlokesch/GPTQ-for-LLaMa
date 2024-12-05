#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=testing_args
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=args_testing_output_%A.out
#SBATCH --error=args_testing_error_%A.out

module purge
module load 2022
module load Anaconda3/2022.05
module load cuDNN/8.6.0.163-CUDA-11.8.0

source ~/.bashrc
cd $HOME/GPTQ-for-LLaMa
conda activate llama

pip uninstall transformers
pip install transformers==4.37.2

pip install toml
pip install triton==2.0.0


llama3_8b="meta-llama/Meta-Llama-3-8B"  # Llama 3 8B

wbits=4
groupsize=128

model_name="llama-3-8b-${wbits}bit-${groupsize}g"
model_file="tensors/${model_name}/${model_name}.pt"
safetensor_file="tensors/${model_name}/${model_name}.safetensors"
save="--save ${model_file}"
save_tensors="--save_safetensors ${safetensor_file}"
save_model="--save_model ${model_name}"
load="--load ${model_file}"
load_save_only="--load_save_only"

echo "Starting with ${wbits} bits and groupsize ${groupsize}."

# Quantize with GPTQ
echo -e "y\n" | python3 llama.py --wbits ${wbits} --groupsize ${groupsize} --eval ${save} ${save_tensors} ${save_model} ${load} ${load_save_only} ${llama3_8b} wikitext2

echo "Done with ${wbits} bits and groupsize ${groupsize}."

