from transformers import AutoModel, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModel.from_pretrained(model_id, cache_dir="weights/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="weights/DeepSeek-R1-Distill-Qwen-1.5B")

model.save_pretrained("weights/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer.save_pretrained("weights/DeepSeek-R1-Distill-Qwen-1.5B")