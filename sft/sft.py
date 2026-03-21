#!/usr/bin/env python3
"""
Venom-R1 — Supervised Fine-Tuning con LoRA

Uso:
  # Single GPU
  python sft.py --config config.yaml

  # Multi GPU (scegli il config accelerate in base alle tue GPU)
  accelerate launch --config_file accelerate_configs/zero3_4gpu.yaml sft.py --config config.yaml

  # Override rapido di un parametro senza modificare il YAML
  python sft.py --config config.yaml --override learning_rate=1e-5 num_train_epochs=5
"""
import argparse
import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig, setup_chat_format

from configs import VenomConfig
from utils.data import get_dataset
from utils.model import get_tokenizer, get_model, get_lora_config
from utils.callbacks import get_callbacks

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def apply_overrides(cfg: VenomConfig, overrides: list[str]) -> VenomConfig:
    """
    Permette di sovrascrivere parametri flat da CLI senza modificare il YAML.
    Esempio: --override learning_rate=1e-5 num_train_epochs=5
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override malformato '{item}'. Usa: chiave=valore")
        key, raw_value = item.split("=", 1)

        # Prova a convertire al tipo corretto
        for cast in (int, float, lambda x: x.lower() == "true" if x.lower() in ("true", "false") else None, str):
            try:
                value = cast(raw_value)
                if value is None:
                    continue
                break
            except (ValueError, TypeError):
                continue

        if not hasattr(cfg, key):
            raise ValueError(f"Parametro '{key}' non esiste in VenomConfig")
        setattr(cfg, key, value)
        logger.info(f"⚙️  Override: {key} = {value}")
    return cfg


def build_sft_config(cfg: VenomConfig) -> SFTConfig:
    """Costruisce SFTConfig da VenomConfig."""

    report_to = ["wandb"] if cfg.wandb.enabled else ["none"]

    # dataset_text_field: per formato "text" serve specificarlo,
    # per "messages" TRL lo gestisce automaticamente via chat_template
    dataset_text_field = "text" if cfg.dataset.format == "text" else "messages"

    return SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        max_seq_length=cfg.max_seq_length,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy=cfg.eval_strategy,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=report_to,
        run_name=cfg.wandb.run_name if cfg.wandb.enabled else None,
        use_liger_kernel=cfg.use_liger_kernel,
        dataset_text_field=dataset_text_field,
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Venom-R1 SFT Training")
    parser.add_argument("--config",   type=str, default="config.yaml",
                        help="Path al file config YAML")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Override parametri: chiave=valore (es. learning_rate=1e-5)")
    args = parser.parse_args()

    setup_logging()

    # ── Carica config ──
    cfg = VenomConfig.from_yaml(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    set_seed(cfg.seed)

    logger.info("=" * 55)
    logger.info("  🐍 Venom-R1 SFT Training")
    logger.info(f"  Modello  : {cfg.model_name}")
    logger.info(f"  Dataset  : {cfg.dataset.name} (format={cfg.dataset.format})")
    logger.info(f"  Output   : {cfg.output_dir}")
    logger.info(f"  LoRA     : r={cfg.lora.r} | alpha={cfg.lora.lora_alpha}")
    logger.info(f"  Sequenza : {cfg.max_seq_length} token")
    logger.info("=" * 55)

    # ── W&B ──
    if cfg.wandb.enabled:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
        if cfg.wandb.entity:
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        logger.info(f"W&B abilitato → project={cfg.wandb.project}")

    # ── Checkpoint detection ──
    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        last_checkpoint = get_last_checkpoint(cfg.output_dir)
        if last_checkpoint:
            logger.info(f"⏩ Checkpoint trovato: {last_checkpoint}")

    # ── Carica componenti ──
    dataset   = get_dataset(cfg.dataset)
    tokenizer = get_tokenizer(cfg)
    model     = get_model(cfg)

    # ── Chat template fallback → ChatML ──
    if tokenizer.chat_template is None:
        logger.info("Nessun chat template → applico ChatML di default")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    # ── Config training ──
    training_args = build_sft_config(cfg)

    # ── LoRA ──
    peft_config = get_lora_config(cfg)

    # ── Trainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=get_callbacks([]),     # passa nomi callback da config se vuoi
    )

    # ── Stampa parametri trainabili ──
    trainer.model.print_trainable_parameters()

    # ──────────────────────────────────────────────
    # TRAINING
    # ──────────────────────────────────────────────
    logger.info("🚀 Avvio training...")
    checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Metriche
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ──────────────────────────────────────────────
    # SALVATAGGIO
    # ──────────────────────────────────────────────
    logger.info(f"💾 Salvataggio modello in: {cfg.output_dir}")

    # Allinea eos_token per evitare generazione infinita con pipeline()
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Ripristina use_cache per inference
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(cfg.output_dir)

    logger.info("✅ Training completato!")

    # ──────────────────────────────────────────────
    # VALUTAZIONE FINALE (opzionale)
    # ──────────────────────────────────────────────
    if cfg.eval_strategy != "no" and "test" in dataset:
        logger.info("📊 Valutazione finale sul test set...")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
