from transformers import AutoModelForCausalLM
import torch

# Load the model
model = AutoModelForCausalLM.from_pretrained("weights/Llama3.2-Instruct", torch_dtype=torch.float16, device_map="auto")

# Print the model architecture
print(model)
