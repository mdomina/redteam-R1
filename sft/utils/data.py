"""
Dataset loading e preprocessing per Venom-R1 SFT.

Formati supportati (configurabili da YAML):
  - messages       → lista di dict {role, content}  →  SFTTrainer applica chat template
  - prompt_response → due colonne →  convertite in messages
  - text           → testo già formattato con template finale
"""
import logging
from datasets import load_dataset, DatasetDict, Dataset
from configs import DatasetConfig

logger = logging.getLogger(__name__)


def get_dataset(cfg: DatasetConfig) -> DatasetDict:
    """
    Carica il dataset da HuggingFace e restituisce un DatasetDict
    con chiavi "train" e opzionalmente "test".
    """
    logger.info(f"📦 Caricamento dataset: {cfg.name} (config={cfg.config}, format={cfg.format})")

    raw = load_dataset(cfg.name, cfg.config)

    # --- Train split ---
    if isinstance(raw, DatasetDict):
        if cfg.train_split not in raw:
            available = list(raw.keys())
            raise ValueError(
                f"Split '{cfg.train_split}' non trovato nel dataset. "
                f"Split disponibili: {available}"
            )
        train_data: Dataset = raw[cfg.train_split]
    else:
        train_data = raw  # Dataset singolo

    # --- Limita campioni (modalità debug) ---
    if cfg.max_samples is not None:
        n = min(cfg.max_samples, len(train_data))
        train_data = train_data.select(range(n))
        logger.info(f"⚠️  max_samples attivo → training su {n} campioni")

    # --- Normalizza formato ---
    train_data = _normalize_format(train_data, cfg)

    # --- Test split ---
    if cfg.test_split and isinstance(raw, DatasetDict) and cfg.test_split in raw:
        test_data = _normalize_format(raw[cfg.test_split], cfg)
        result = DatasetDict({"train": train_data, "test": test_data})

    elif cfg.test_size and cfg.test_size > 0:
        splits = train_data.train_test_split(test_size=cfg.test_size, seed=42)
        result = DatasetDict({"train": splits["train"], "test": splits["test"]})

    else:
        result = DatasetDict({"train": train_data})

    _log_dataset_info(result)
    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _normalize_format(dataset: Dataset, cfg: DatasetConfig) -> Dataset:
    """
    Porta il dataset al formato atteso da SFTTrainer:
      - messages format  → colonna "messages" con [{role, content}, ...]
      - prompt_response  → converte in messages
      - text             → colonna "text" con stringa già formattata
    """
    fmt = cfg.format

    if fmt == "messages":
        if cfg.messages_column not in dataset.column_names:
            raise ValueError(
                f"Colonna '{cfg.messages_column}' non trovata nel dataset. "
                f"Colonne disponibili: {dataset.column_names}"
            )
        # Rinomina se serve
        if cfg.messages_column != "messages":
            dataset = dataset.rename_column(cfg.messages_column, "messages")
        # Rimuovi colonne extra per pulizia
        cols_to_remove = [c for c in dataset.column_names if c != "messages"]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)
        return dataset

    elif fmt == "prompt_response":
        for col in [cfg.prompt_column, cfg.response_column]:
            if col not in dataset.column_names:
                raise ValueError(
                    f"Colonna '{col}' non trovata. Colonne disponibili: {dataset.column_names}"
                )

        def to_messages(example):
            return {
                "messages": [
                    {"role": "user",      "content": example[cfg.prompt_column]},
                    {"role": "assistant", "content": example[cfg.response_column]},
                ]
            }

        dataset = dataset.map(
            to_messages,
            remove_columns=dataset.column_names,
            desc="Conversione prompt_response → messages",
        )
        return dataset

    elif fmt == "text":
        if cfg.text_column not in dataset.column_names:
            raise ValueError(
                f"Colonna '{cfg.text_column}' non trovata. "
                f"Colonne disponibili: {dataset.column_names}"
            )
        if cfg.text_column != "text":
            dataset = dataset.rename_column(cfg.text_column, "text")
        cols_to_remove = [c for c in dataset.column_names if c != "text"]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)
        return dataset

    else:
        raise ValueError(
            f"Formato '{fmt}' non supportato. Usa: messages | prompt_response | text"
        )


def _log_dataset_info(ds: DatasetDict):
    parts = []
    for split, data in ds.items():
        parts.append(f"{split}: {len(data):,} esempi")
    logger.info(f"✅ Dataset pronto — {' | '.join(parts)}")
