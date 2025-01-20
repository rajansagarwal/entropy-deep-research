from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def plot_entropy_metrics(entropies, varentropies, output_text, tokenizer, save_path='entropy_plot.png'):
    tokens = tokenizer.tokenize(output_text)
    tokens = tokens[:len(entropies)]
    tokens = [t.replace('Ġ', '').replace('Ċ', '\n') for t in tokens]

    plt.figure(figsize=(20, 8))
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