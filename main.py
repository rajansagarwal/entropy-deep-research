from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def run_inference(prompt, device=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained("weights/Llama3.2-Instruct", torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("weights/Llama3.2-Instruct", torch_dtype=torch.float16, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    inputs = {key: value.to(device) for key, value in inputs.items()}

    generate_ids = model.generate(**inputs, max_length=30)

    output_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    return output_text

prompt = "Hey, are you conscious? Can you talk to me?"
run_inference(prompt=prompt)