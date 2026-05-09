import os

# Model Config
LLM_PROVIDER = "ollama"
MODEL_NAME = "llama3:latest"  # Or whatever model you downloaded

# Output
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True) 