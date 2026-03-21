"""
Configurazione Venom-R1 SFT
Tutti i parametri sono definiti qui e caricati da config.yaml
"""
from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class LoRAConfig:
    """Parametri LoRA / PEFT"""
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"


@dataclass
class DatasetConfig:
    """
    Configurazione dataset HuggingFace.

    Formati supportati:
        messages       → colonna con lista [{role, content}, ...]  (ChatML nativo)
        prompt_response → due colonne separate prompt + response
        text           → colonna con testo già formattato con template
    """
    name: str = ""
    config: Optional[str] = None           # subset HF (es. "default", "all")
    format: str = "messages"               # messages | prompt_response | text

    # Nomi colonne (usati in base al formato scelto)
    messages_column: str = "messages"
    prompt_column: str = "prompt"
    response_column: str = "response"
    text_column: str = "text"

    train_split: str = "train"
    test_split: Optional[str] = None       # se null → split automatico da train
    test_size: float = 0.05               # % test se test_split è null

    max_samples: Optional[int] = None     # null = tutto; intero = limite (utile per debug)


@dataclass
class WandbConfig:
    """Configurazione Weights & Biases"""
    enabled: bool = False
    project: str = "venom-r1"
    entity: Optional[str] = None          # null = account HF default
    run_name: Optional[str] = None        # null = nome auto-generato


@dataclass
class VenomConfig:
    """
    Configurazione principale — specchio 1:1 di config.yaml
    """

    # ---------- Modello ----------
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model_revision: str = "main"
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"
    torch_dtype: str = "bfloat16"          # bfloat16 | float16 | float32 | auto

    # ---------- Dataset ----------
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # ---------- LoRA ----------
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # ---------- Training ----------
    output_dir: str = "./output/venom-r1"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_length: int = 8192
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    eval_strategy: str = "steps"           # no | steps | epoch
    save_total_limit: int = 3
    seed: int = 42
    use_liger_kernel: bool = False         # True = più veloce (richiede pip install liger-kernel)

    # ---------- W&B ----------
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "VenomConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "VenomConfig":
        cfg = cls()
        _nested = {
            "dataset": (DatasetConfig, "dataset"),
            "lora":    (LoRAConfig,    "lora"),
            "wandb":   (WandbConfig,   "wandb"),
        }
        for key, value in d.items():
            if key in _nested:
                klass, attr = _nested[key]
                if isinstance(value, dict):
                    valid = {k: v for k, v in value.items()
                             if k in klass.__dataclass_fields__}
                    setattr(cfg, attr, klass(**valid))
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise ValueError(f"Parametro sconosciuto nel config: '{key}'")
        return cfg
