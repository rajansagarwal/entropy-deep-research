from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def calculate_entropy_varentropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)  # Avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    varentropy = torch.sum(probs * (log_probs + entropy.unsqueeze(-1))**2, dim=-1)
    return entropy, varentropy

def create_chatbot_prompt(user_input):
    return f"""You are a helpful AI assistant. Please respond to the following message:
Human: ball light green light squeeze light light light light light light
Assistant:"""

def plot_entropy_metrics(entropies, varentropies, output_text, tokenizer, save_path='entropy_plot.png'):
    tokens = tokenizer.tokenize(output_text)
    tokens = tokens[:len(entropies)]
    tokens = [t.replace('Ġ', '').replace('Ċ', '\n') for t in tokens]

    plt.figure(figsize=(20, 8))
    
    # Create two subplots
    plt.subplot(1, 2, 1)
    plt.plot(range(len(entropies)), entropies, label='Entropy', color='blue', linewidth=2)
    plt.title('Token-wise Entropy', fontsize=14, pad=20)
    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.xticks(range(len(tokens)), tokens, rotation=90, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(varentropies)), varentropies, label='Varentropy', color='red', linewidth=2)
    plt.title('Token-wise Varentropy', fontsize=14, pad=20)
    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel('Varentropy', fontsize=12)
    plt.xticks(range(len(tokens)), tokens, rotation=90, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path)
    plt.close()

def run_inference_with_entropy(prompt, past_n=5, device=None):
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
    
    # Format prompt as chatbot conversation
    chatbot_prompt = create_chatbot_prompt(prompt)
    
    # Inference parameters
    temperature = 0.7
    top_p = 0.9
    top_k = 50
    max_new_tokens = 100 
    max_input_length = 512
    
    inputs = tokenizer(chatbot_prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True,
        output_attentions=True
    )
    
    logits = torch.stack(outputs.scores, dim=1).to(torch.float32)

    attention_tensors = [attn[0] for attn in outputs.attentions]
    attention_scores = [attn[0].to(torch.float32) for attn in outputs.attentions]

    head_scores = attention_scores
    head_scores = attention_scores[0][0][0].cpu().detach().numpy()  # [B, L, H]
    
    print(f"Attention Scores: {head_scores}")

    entropy_values = []
    varentropy_values = []
    past_n_logits = []
    
    for i, token_logits in enumerate(logits[0]):
        past_n_logits.append(token_logits)
        if len(past_n_logits) > past_n:
            past_n_logits.pop(0)
        
        if len(past_n_logits) >= past_n:
            stacked_logits = torch.stack(past_n_logits, dim=0)
            entropy, varentropy = calculate_entropy_varentropy(stacked_logits)
            entropy_values.append(entropy.mean().item())
            varentropy_values.append(varentropy.mean().item())
    
    # Decode output and extract assistant's response
    generated_ids = outputs.sequences
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Plot and save entropy metrics
    plot_entropy_metrics(entropy_values, varentropy_values, full_text, tokenizer)
    
    # Extract only the assistant's response
    response = full_text.split("Assistant:")[-1].strip()
    
    return response, entropy_values, varentropy_values

def chat_with_entropy():
    print("Chat with the AI Assistant (type 'exit' to end)")
    while True:
        """         
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        """
        response, entropies, varentropies = run_inference_with_entropy(
            prompt="",
            past_n=5,
            device="cpu"
        )
        
        print(f"\nAssistant: {response}")
        print("\nEntropy metrics have been saved to 'entropy_plot.png'")
        break

if __name__ == "__main__":
    chat_with_entropy()