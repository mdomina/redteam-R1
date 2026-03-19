"""
Pipeline principale di training GRPO statico.

Flusso:
    1. Carica i dati di pentest step-by-step da ./dataset/*
    2. Carica il modello DeepSeek-R1-0528-Qwen3-8B con Unsloth (ottimizzato per velocità)
    3. Applica LoRA (PEFT) per il fine-tuning efficiente (solo ~1-2% dei parametri)
    4. Filtra i campioni troppo lunghi (> 95° percentile) per evitare OOM
    5. Calcola max_prompt_length e max_completion_length dinamicamente
    6. Inizializza SwanLab per il monitoraggio del training
    7. Addestra con GRPOTrainer usando le due reward functions (format + accuracy)
    8. Salva il modello finale
"""
import os

import swanlab
import numpy as np
from unsloth import FastLanguageModel
from trl import GRPOTrainer
from utils.load_dataset import load_ctf_data
from utils.trainer_config import create_training_config
from utils.reward import accuracy_ctf_reward, format_reward
from dataclasses import asdict

# Cartella con i file JSON degli step di pentest (es. ./dataset/ctf/session_001.json)
step_folder = "./dataset/ctf"

# Dove salvare i checkpoint e il modello finale
output_dir = "./model/grpo_stage1"

# Lunghezza massima della sequenza (prompt + risposta) in token.
# Limita la memoria GPU e definisce il contesto massimo del modello.
max_seq_length = 8192

# --- 1. Caricamento dataset ---
dataset = load_ctf_data(step_folder)

# Larger rank = smarter, but slower
lora_rank = 32  

# --- 2. Caricamento modello base ---
# Unsloth ottimizza il modello per inferenza rapida e training efficiente.
# load_in_4bit=False → precisione piena (bf16), nessuna quantizzazione
# fast_inference=True → abilita le ottimizzazioni di Unsloth per vLLM-like speed
# gpu_memory_utilization=0.7 → usa il 70% della VRAM disponibile
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    load_in_4bit=True, # False for LoRA 16bit
    fast_inference=True, # vLLM durante il training
    gpu_memory_utilization=0.7, # Reduce if out of memory
    device_map="auto"
)
# --- 3. Applicazione LoRA (PEFT) ---
# Aggiunge adattatori LoRA ai layer di attenzione e MLP del transformer.
# r=32: rango della decomposizione LoRA (più alto = più parametri, più capacità)
# lora_alpha=64: scala i gradienti LoRA (solitamente 2x il rango)
# target_modules: i layer su cui applicare LoRA
#   - q_proj, k_proj, v_proj, o_proj → layer di attenzione (Q, K, V, Output)
#   - gate_proj, up_proj, down_proj  → layer MLP (architettura SwiGLU)
# use_gradient_checkpointing="unsloth" → risparmia VRAM ricalcolando le attivazioni
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha = 64,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)
# --- 4. Filtraggio campioni per lunghezza ---
# Tokenizza i prompt per misurarne la lunghezza in token.
tokenized = dataset.map(
    lambda x: {"tokens" : tokenizer.apply_chat_template(x["prompt"], add_generation_prompt = True, tokenize = True)},
    batched = True,
)
# Stampa un esempio di prompt tokenizzato per debug visivo
print(tokenizer.decode(tokenized[0]["tokens"]))
tokenized = tokenized.map(lambda x: {"L" : len(x["tokens"])})

# Usa il 95° percentile come lunghezza massima: scarta solo il 5% più lungo
# evitando OOM causati da outlier molto lunghi.
maximum_length = int(np.quantile(tokenized["L"], 0.95))
print("Max Length = ", maximum_length)

# Filtra il dataset tenendo solo i campioni entro la lunghezza massima
dataset = dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized  # libera memoria

# --- 5. Calcolo lunghezze prompt/completion ---
# max_prompt_length: lunghezza massima dell'input (+1 per sicurezza)
# max_completion_length: spazio rimanente per la risposta del modello
max_prompt_length = maximum_length + 1
# Non lasciare che superi i 1024 o 2048 token di risposta
max_completion_length = min(2048, max_seq_length - max_prompt_length)

training_args = create_training_config(max_prompt_length, max_completion_length, output_dir, tokenizer)

# --- 6. Inizializzazione SwanLab (monitoring) ---
# Traccia loss, reward, LR e altre metriche durante il training.
swanlab.init(
    project="pentest-r1",
    experiment_name="grpo-stage1",
    config=asdict(training_args),
    # mode="disabled",  # Decommenta per disabilitare il logging (run offline)
)

# --- 7. Training GRPO ---
# GRPOTrainer implementa Group Relative Policy Optimization:
#   - genera num_generations=4 risposte per ogni prompt
#   - calcola le reward con format_reward e accuracy_reward
#   - aggiorna i pesi del modello per massimizzare le reward relative al gruppo
#   - questo è l'approccio usato da DeepSeek-R1 per il ragionamento
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward, accuracy_ctf_reward],
    args=training_args,
    train_dataset=dataset,
)

trainer.train(resume_from_checkpoint=True)

# --- 8. Salvataggio modello finale ---
final_model_dir = os.path.join(output_dir, "final_model")
os.makedirs(final_model_dir, exist_ok=True)
print(f"Saving final model to {final_model_dir} ...")

model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print("Model and tokenizer saved successfully.")