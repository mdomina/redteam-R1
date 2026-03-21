"""
Caricamento modello, tokenizer e configurazione LoRA per Venom-R1 SFT.
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from configs import VenomConfig

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
    "auto":     "auto",
}


def get_tokenizer(cfg: VenomConfig) -> AutoTokenizer:
    logger.info(f"🔤 Caricamento tokenizer: {cfg.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        trust_remote_code=cfg.trust_remote_code,
    )

    # Assicura che pad_token sia impostato (richiesto da SFTTrainer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token non trovato → impostato uguale a eos_token")

    logger.info(f"Vocab size: {tokenizer.vocab_size:,} | EOS: '{tokenizer.eos_token}'")
    return tokenizer


def get_model(cfg: VenomConfig) -> AutoModelForCausalLM:
    logger.info(f"🤖 Caricamento modello: {cfg.model_name}")

    torch_dtype = DTYPE_MAP.get(cfg.torch_dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch_dtype,
        # use_cache incompatibile con gradient_checkpointing
        use_cache=not cfg.gradient_checkpointing,
    )

    # Conta parametri totali
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parametri totali: {total_params / 1e9:.2f}B")

    return model


def get_lora_config(cfg: VenomConfig) -> LoraConfig:
    lc = cfg.lora
    logger.info(
        f"🔧 LoRA — r={lc.r} | alpha={lc.lora_alpha} | "
        f"dropout={lc.lora_dropout} | modules={lc.target_modules}"
    )
    return LoraConfig(
        r=lc.r,
        lora_alpha=lc.lora_alpha,
        lora_dropout=lc.lora_dropout,
        target_modules=lc.target_modules,
        bias=lc.bias,
        task_type="CAUSAL_LM",
    )
