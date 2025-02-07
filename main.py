from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import warnings

# Suppress SSL warnings
warnings.filterwarnings('ignore', category=UserWarning)

def create_chatbot_prompt(user_input):
    return f"User: {user_input.strip()}\nAssistant: "

def calculate_entropy_varentropy(logits):
    """
    Compute entropy and varentropy from logits (assumed shape [vocab_size]).
    """
    logits = logits.float()
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs)
    varentropy = torch.sum(probs * (log_probs + entropy)**2)
    return entropy.item(), varentropy.item()

def run_inference_with_flags(prompt, past_n=5, device=None):
    # Uncertainty thresholds (tweak these as needed)
    threshold_entropy = 2.5  
    threshold_varentropy = 0.7  

    # Generation parameters
    max_new_tokens = 300
    temperature = 0.7
    top_p = 0.3
    max_input_length = 512

    if device is None:
        device = torch.device("cpu")

    # Load model and tokenizer (using your working setup)
    model = AutoModelForCausalLM.from_pretrained(
        "weights/Llama3.2-Instruct", 
        torch_dtype=torch.float16, 
        device_map=device,
        pad_token_id=None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "weights/Llama3.2-Instruct",
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    chatbot_prompt = create_chatbot_prompt(prompt)
    inputs = tokenizer(
        chatbot_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=max_input_length
    ).to(device)

    # Use generate with output_scores and return_dict_in_generate for post-processing
    output = model.generate(
        **inputs,
        max_length=max_input_length + max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True
    )
    
    # Decode the final generated text (excluding the prompt)
    input_length = inputs.input_ids.shape[1]
    generated_ids = output.sequences[0][input_length:]
    final_response = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Post-process the output scores to record flag events.
    # output.scores is a tuple (length = number of generated tokens) of logits [1, vocab_size].
    scores = output.scores
    flag_events = []
    sliding_entropies = []
    sliding_varentropies = []
    last_flag = None

    for step, logits in enumerate(scores):
        ent, varent = calculate_entropy_varentropy(logits[0])
        sliding_entropies.append(ent)
        sliding_varentropies.append(varent)
        if len(sliding_entropies) >= past_n:
            avg_ent = np.mean(sliding_entropies[-past_n:])
            avg_varent = np.mean(sliding_varentropies[-past_n:])
            # Determine flag based on averages.
            if avg_ent <= threshold_entropy and avg_varent > threshold_varentropy:
                current_flag = "<branch>"
            elif avg_ent > threshold_entropy and avg_varent <= threshold_varentropy:
                current_flag = "<think>"
            elif avg_ent > threshold_entropy and avg_varent > threshold_varentropy:
                current_flag = "<resample>"
            else:
                current_flag = None

            # Record the flag only if it changes.
            if current_flag is not None and current_flag != last_flag:
                flag_events.append((step, current_flag, avg_ent, avg_varent))
                last_flag = current_flag
            elif current_flag is None:
                last_flag = None

    return final_response, flag_events

if __name__ == "__main__":
    prompt = ("User: Which is greater? 9.9 or 9.11?\n"
              "Assistant: ")
    response, flags = run_inference_with_flags(prompt, past_n=5, device="cpu")
    print("Final Response:\n", response)
    if flags:
        print("\nFlag events during generation:")
        for step, flag, avg_ent, avg_varent in flags:
            print(f" Step {step}: {flag} (avg_entropy={avg_ent:.3f}, avg_varentropy={avg_varent:.3f})")
