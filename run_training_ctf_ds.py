"""
Pipeline di training GRPO con transformers + DeepSpeed ZeRO-2 (senza Unsloth).

Flusso:
    1. Carica i dati di pentest step-by-step da ./dataset/*
    2. Carica il modello con AutoModelForCausalLM (bf16, device_map="auto")
    3. Applica LoRA (PEFT) per il fine-tuning efficiente
    4. Filtra i campioni troppo lunghi (> 95° percentile) per evitare OOM
    5. Calcola max_prompt_length e max_completion_length dinamicamente
    6. Inizializza SwanLab per il monitoraggio del training
    7. Addestra con GRPOTrainer + DeepSpeed ZeRO-2 con offload optimizer su CPU
    8. Salva il modello finale
"""
import os

import swanlab
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer
from utils.load_dataset import load_ctf_data
from utils.reward import accuracy_ctf_reward, format_reward
from dataclasses import asdict

# Cartella con i file JSON degli step di pentest (es. ./dataset/ctf/session_001.json)
step_folder = "./dataset/ctf"

# Dove salvare i checkpoint e il modello finale
output_dir = "./model/grpo_stage1_ds"

# Lunghezza massima della sequenza (prompt + risposta) in token.
max_seq_length = 8192

# Larger rank = smarter, but slower
lora_rank = 32

# --- 1. Caricamento dataset ---
dataset = load_ctf_data(step_folder)

# --- 2. Caricamento modello base ---
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# --- 3. Applicazione LoRA (PEFT) ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=lora_rank,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.0,
    bias="none",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Abilita gradient checkpointing per risparmiare VRAM
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# --- 4. Filtraggio campioni per lunghezza ---
tokenized = dataset.map(
    lambda x: {"tokens": tokenizer.apply_chat_template(x["prompt"], add_generation_prompt=True, tokenize=True)},
    batched=True,
)
print(tokenizer.decode(tokenized[0]["tokens"]))
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})

# Usa il 95° percentile come lunghezza massima
maximum_length = int(np.quantile(tokenized["L"], 0.95))
print("Max Length = ", maximum_length)

dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

# --- 5. Calcolo lunghezze prompt/completion ---
max_prompt_length = maximum_length + 1
max_completion_length = min(2048, max_seq_length - max_prompt_length)

# --- 6. Configurazione training con DeepSpeed ---
training_args = GRPOConfig(
    temperature=0.6,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    bf16=True,
    logging_steps=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    num_generations=8,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    save_steps=10,
    save_total_limit=3,
    num_train_epochs=1,
    report_to="swanlab",
    output_dir=output_dir,
    deepspeed="./deepspeed_zero2_offload.json",
)

# --- 7. Inizializzazione SwanLab (monitoring) ---
swanlab.init(
    project="pentest-r1",
    experiment_name="grpo-stage1-ds",
    config=asdict(training_args),
)

# --- 8. Training GRPO ---
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward, accuracy_ctf_reward],
    args=training_args,
    train_dataset=dataset,
)

trainer.train(resume_from_checkpoint=True)

# --- 9. Salvataggio modello finale ---
final_model_dir = os.path.join(output_dir, "final_model")
os.makedirs(final_model_dir, exist_ok=True)
print(f"Saving final model to {final_model_dir} ...")

model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print("Model and tokenizer saved successfully.")
