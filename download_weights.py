from transformers import AutoModel, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModel.from_pretrained(model_id, cache_dir="weights/Llama3.2-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="weights/Llama3.2-Instruct")

model.save_pretrained("weights/Llama3.2-Instruct")
tokenizer.save_pretrained("weights/Llama3.2-Instruct")