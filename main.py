import ollama

model_name = "qwen3:4b-instruct"
prompt = "how do i make a cake"

response = ollama.generate(
    model=model_name,
    prompt=prompt,
    options={
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 500
    }
)

print(response.response)
