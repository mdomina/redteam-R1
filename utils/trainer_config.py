from trl import GRPOConfig

def create_training_config(max_prompt_length, max_completion_length, output_dir, tokenizer):
    """
    Crea e restituisce la configurazione GRPOConfig per il training.

    Parametri chiave:
      temperature=0.6            → campionamento deterministico-moderato durante il rollout
      learning_rate=5e-6         → LR bassa, adatta al fine-tuning con RL
      weight_decay=0.01          → regolarizzazione L2 lieve
      warmup_ratio=0.03          → 3% degli step usati per il warmup del LR
      lr_scheduler_type="cosine" → decay co-sinusoidale del learning rate
      optim="adamw_torch"        → ottimizzatore AdamW nativo PyTorch
      bf16=True                  → bfloat16 per efficienza su GPU moderne (A100/H100)
      per_device_train_batch_size=1  → batch size 1 per device (memoria limitata)
      gradient_accumulation_steps=32 → batch effettivo = 1*32 = 32 campioni per update
      num_generations=4          → GRPO genera 4 risposte per prompt e le confronta
      save_steps=50              → salva checkpoint ogni 50 step
      save_total_limit=3         → mantiene solo gli ultimi 3 checkpoint
      num_train_epochs=2         → 2 epoche sul dataset
      report_to="swanlab"        → logging su SwanLab (alternativa a W&B)
    """
    training_args = GRPOConfig(
        temperature=0.6,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",# adamw_8bit: quantizzato; adamw_torch: normale
        bf16=True,
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        save_steps=50,
        save_total_limit=3,
        num_train_epochs=1,
        report_to="swanlab",
        output_dir=output_dir,
    )

    return training_args