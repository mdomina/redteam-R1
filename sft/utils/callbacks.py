"""
Callbacks per Venom-R1 SFT.
Aggiungi qui i tuoi callback custom e registrali nel dizionario CALLBACKS.
"""
import logging
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class LogSamplesCallback(TrainerCallback):
    """
    Logga il numero di token processati ad ogni checkpoint.
    Utile per monitorare il throughput del training.
    """
    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            loss = logs.get("loss", "N/A")
            lr   = logs.get("learning_rate", "N/A")
            logger.info(f"Step {step:>6} | loss={loss:.4f} | lr={lr:.2e}")


# ------------------------------------------------------------------
# Registry — aggiungi qui i tuoi callback custom
# Formato: "nome_stringa": ClasseCallback
# ------------------------------------------------------------------
CALLBACKS = {
    "log_samples": LogSamplesCallback,
}


def get_callbacks(callback_names: list) -> list:
    """
    Istanzia i callback richiesti dalla lista di nomi.
    I nomi devono essere chiavi nel dizionario CALLBACKS.
    """
    result = []
    for name in callback_names:
        if name not in CALLBACKS:
            raise ValueError(
                f"Callback '{name}' non trovato. "
                f"Disponibili: {list(CALLBACKS.keys())}"
            )
        result.append(CALLBACKS[name]())
        logger.info(f"Callback caricato: {name}")
    return result
