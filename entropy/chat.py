from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from entropy.inference import run_inference_with_entropy

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