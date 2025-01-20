from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

def run_inference_with_entropy(prompt, device=None):
    # Determine device
    if device is None:
        device = "cpu"
    # Load model and tokenizer
    model_path = "weights/Llama3.2-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Handle tokenizer pad token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Inference parameters
    temperature = 0.7
    top_p = 0.9
    top_k = 50
    max_new_tokens = 50
    max_input_length = 512
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate text and logits
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    # Decode output
    generated_ids = outputs.sequences
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Fetch logits for entropy and varentropy calculation
    logits = torch.stack(outputs.scores, dim=1).to(torch.float32)  # Convert to float32
    print("Logits shape:", logits.shape)
    
    # Function to calculate entropy and varentropy for a single token
    def calculate_entropy_varentropy_single_token(logit):
        prob = F.softmax(logit, dim=-1)
        # Ensure probabilities sum to 1
        prob = prob / prob.sum()
        log_prob = torch.log(prob + 1e-10)
        entropy = -torch.sum(prob * log_prob)
        varentropy = torch.sum(prob * (log_prob + entropy)**2)
        return entropy.item(), varentropy.item()
    
    # Calculate entropy and varentropy for each token
    entropies = []
    varentropies = []
    for token_logits in logits[0]:  # Assuming batch_size=1
        entropy, varentropy = calculate_entropy_varentropy_single_token(token_logits)
        entropies.append(entropy)
        varentropies.append(varentropy)
    
    # Print the values
    for i, (ent, var) in enumerate(zip(entropies, varentropies)):
        print(f"Token {i}: Entropy={ent:.4f}, Varentropy={var:.4f}")
    
    print("Generated Text:", output_text)
    
    return output_text

# Example prompt
prompt = "Who are you?"
run_inference_with_entropy(prompt=prompt)