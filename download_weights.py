from transformers import AutoModel

model_id = "meta-llama/Llama-3.2-1B"
model = AutoModel.from_pretrained(model_id, cache_dir="weights/Llama3.2")