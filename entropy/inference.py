from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
import warnings
import sys  # For printing flag events to stderr

from plot import plot_entropy_metrics

def create_chatbot_prompt(user_input):
    return f"User: {user_input.strip()}\nAssistant: "

def calculate_entropy_varentropy(logits):
    """
    Compute the entropy and varentropy from logits of shape [N, vocab_size].
    """
    logits = logits.float()
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    varentropy = torch.sum(probs * (log_probs + entropy.unsqueeze(-1))**2, dim=-1)
    return entropy, varentropy

def run_inference_with_streaming(prompt, past_n=5, device=None):
    # Set thresholds (these can be tuned)
    threshold_entropy = 2.5  
    threshold_varentropy = 0.7  

    repetition_penalty = 1.2
    max_repetitions = 3
    max_new_tokens = 300
    flag_cooldown_duration = 10  # tokens to wait before printing a new flag

    # Open log file for entropy tracking
    with open("entropy_log.txt", "w") as log_file:
        log_file.write(f"Entropy Log for Prompt: {prompt}\n" + "-" * 50 + "\n")

    if device is None:
        device = torch.device("cpu")

    model_path = "weights/Llama3.2-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    chatbot_prompt = create_chatbot_prompt(prompt)
    base_temperature = 0.7
    # We set top_p to a low value only if sampling is used, but default will be greedy.
    top_p = 0.3  
    max_input_length = 512

    inputs = tokenizer(
        chatbot_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_input_length
    ).to(device)

    entropy_values = []
    varentropy_values = []
    sliding_entropy = []

    # (Optional) add special tokens for flagging (they're not appended to output)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "<reason>" not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": ["<reason>", "</reason>"]})
            model.resize_token_embeddings(len(tokenizer))
    eos_token_id = tokenizer.eos_token_id

    generated_ids = inputs.input_ids
    past_key_values = None
    recent_tokens = []

    flag_events = []
    last_flag = None
    flag_cooldown = 0

    print("\nGenerated response:", end=" ")

    for step in range(max_new_tokens):
        if len(recent_tokens) > 10:
            if recent_tokens.count(recent_tokens[-1]) > max_repetitions:
                print("\n[System: Stopping repetition]", file=sys.stderr)
                break
            recent_tokens.pop(0)

        outputs = model(input_ids=generated_ids, past_key_values=past_key_values, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        entropy, varentropy = calculate_entropy_varentropy(next_token_logits)
        current_entropy = entropy.detach().cpu().numpy()
        current_varentropy = varentropy.detach().cpu().numpy()
        entropy_values.append(current_entropy)
        varentropy_values.append(current_varentropy)
        sliding_entropy.append(current_entropy)

        # Compute sliding averages once we have enough tokens
        if len(sliding_entropy) >= past_n:
            avg_entropy = np.mean(sliding_entropy[-past_n:])
            avg_varentropy = np.mean(varentropy_values[-past_n:])
            with open("entropy_log.txt", "a") as f:
                f.write(f"Window at step {step}: avg_entropy={avg_entropy:.3f}, avg_varentropy={avg_varentropy:.3f}\n")
            # Determine flag based on averages
            if avg_entropy <= threshold_entropy and avg_varentropy > threshold_varentropy:
                flag = "<branch>"
            elif avg_entropy > threshold_entropy and avg_varentropy <= threshold_varentropy:
                flag = "<think>"
            elif avg_entropy > threshold_entropy and avg_varentropy > threshold_varentropy:
                flag = "<resample>"
            else:
                flag = None

            # Only print a new flag if cooldown is zero and flag has changed.
            if flag is not None and flag != last_flag and flag_cooldown == 0:
                flag_message = f"[Flag: {flag} (avg_entropy={avg_entropy:.3f}, avg_varentropy={avg_varentropy:.3f})]"
                print("\n" + flag_message)
                flag_events.append((step, flag, avg_entropy, avg_varentropy))
                last_flag = flag
                flag_cooldown = flag_cooldown_duration
            elif flag is None:
                last_flag = None

        if flag_cooldown > 0:
            flag_cooldown -= 1

        # Get probability distribution.
        probs = F.softmax(next_token_logits, dim=-1)

        # Adaptive decoding based on flag:
        # If uncertainty is high (<think> or <resample>), use greedy (argmax).
        # If moderate (<branch>), lower temperature and sample.
        next_token = torch.argmax(probs, dim=-1, keepdim=True)

        # (Optional) Apply repetition penalty on logits if needed.
        if recent_tokens:
            for token_id in set(recent_tokens):
                next_token_logits[0, token_id] /= repetition_penalty

        recent_tokens.append(next_token.item())
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        token_text = tokenizer.decode(
            next_token[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        if token_text.strip():
            print(token_text, end=" ", flush=True)

        if next_token.item() == eos_token_id:
            break

    if step == max_new_tokens - 1:
        print("\n[Info: Maximum token generation reached. Stopping generation.]", file=sys.stderr)

    response = tokenizer.decode(
        generated_ids[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()

    return response, entropy_values, varentropy_values, flag_events

if __name__ == "__main__":
    prompt = ("Show me the answer to the multiplication of 181 * 223?\n")
    print("\nHuman:", prompt)
    response, entropies, varentropies, flags = run_inference_with_streaming(prompt, past_n=10, device="cpu")
    print("\n\nFinal Response:\n", response)
    if flags:
        print("\nFlag events during generation:")
        for event in flags:
            step, flag, avg_ent, avg_varent = event
            print(f" Step {step}: {flag} (avg_entropy={avg_ent:.3f}, avg_varentropy={avg_varent:.3f})")
