from utils.load_dataset import load_ctf_data_hf
from unsloth import FastLanguageModel
import numpy as np

# Solo tokenizer, no modello — velocissimo
_, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=512,   # non importa per il tokenizer
    load_in_4bit=True,
)

dataset = load_ctf_data_hf("./dataset/ctf")
lengths = [
    len(tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True))
    for x in dataset
]
lengths = np.array(lengths)

print(f"Campioni totali: {len(lengths)}")
print(f"Min:     {lengths.min()}")
print(f"Media:   {lengths.mean():.0f}")
print(f"Mediana: {np.median(lengths):.0f}")
print(f"P95:     {np.quantile(lengths, 0.95):.0f}")
print(f"Max:     {lengths.max()}")