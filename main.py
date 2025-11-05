import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 🧹 Optional: silence Windows symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 🧠 Optional: improve performance for Xet storage
# (uncomment this line if you installed it)
# pip install "huggingface_hub[hf_xet]"

# ============================================================
# ⚙️ Model setup
model_name = "Qwen/Qwen1.5-0.5B"  # open-source version

print(f"🚀 Loading model: {model_name}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# ============================================================
# 🗣️ Prompt
prompt = "Explain in simple terms how large language models work."

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("🧩 Generating response...")

# Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print result
print("\n=============================")
print("🧠 Qwen 1.5-0.5B says:\n")
print(response)
print("=============================\n")