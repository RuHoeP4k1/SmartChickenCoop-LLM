import ollama

# ============================================================
# ⚙️ Model setup
model_name = "qwen:3-4b"  # Name as installed in Ollama

print(f"🚀 Loading model: {model_name}")

# ============================================================
# 🗣️ Prompt
prompt = "Why do cows produce milk?"

print("🧩 Generating response...")

# Generate using Ollama
response = ollama.generate(
    model=model_name,
    prompt=prompt,
    options={
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 200
    }
)

# ============================================================
# 📤 Print result
print("\n=============================")
print(f"🧠 {model_name} says:\n")
print(response["response"])
print("=============================\n")
