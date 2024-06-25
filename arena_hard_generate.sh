#!/bin/bash

# Define an array of model names
models=(
  "allenai/OLMo-1B-hf"
  "allenai/OLMo-7B-hf"
  "CohereForAI/aya-101"
  "google/gemma-7b-it"
  "google/gemma-7b"
  "google/gemma-2b-it"
  "google/gemma-2b"
  "google/gemma-1.1-7b-it"
  "google/gemma-1.1-2b-it"
  "NousResearch/Meta-Llama-3-8B"
)

# Loop through each model and execute the Python script
for model in "${models[@]}"; do
  python generate.py --model_name "$model" --output_path 'arena_hard'
done
