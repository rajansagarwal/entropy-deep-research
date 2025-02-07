# load the model
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("weights/Llama3.2-Instruct", torch_dtype=torch.float16, device_map="cpu")

# load the tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("weights/Llama3.2-Instruct", device_map="cpu")

# inference the model

prompt = "Hey, are you conscious? Can you talk to me?"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
