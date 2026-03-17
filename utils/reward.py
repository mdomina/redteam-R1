import re


def accuracy_reward(prompts, completions, answer, **kwargs):
    """
    Reward function per l'ACCURATEZZA del contenuto generato.
    Confronta la risposta del modello con il ground truth (answer).
    Opera solo sulla parte DOPO </think> (risposta visibile).

    Punteggio massimo: 1.2 (step match + comando esatto)
      +0.2  se il numero dello step generato corrisponde al ground truth
      +1.0  se il comando generato è identico al ground truth
      +similarity*0.7  se il comando è simile (Jaccard > 0.5) ma non identico
                        (ricompensa parziale per comandi quasi corretti)

    Se la risposta non ha </think> o il ground truth non ha un comando,
    lo score rimane 0.0 per quel campione.
    """
    # Cattura tutto ciò che viene dopo </think>: la risposta "visibile" effettiva del modello.
    POST_THINK_PATTERN = re.compile(r"</think>(.*)", re.DOTALL)
    # Estrae il numero dello step da "=== Step N ===" per confrontarlo con il ground truth.
    STEP_NUM_PATTERN = re.compile(r"===\s*Step\s*(\d+)\s*===", re.IGNORECASE)
    # Estrae il comando esatto dalla riga "Command: <cmd>".
    COMMAND_PATTERN = re.compile(r"Command:\s*(.*?)(?:\n|$)", re.DOTALL | re.IGNORECASE)

    rewards = []
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        true_answer = answer[i]
        score = 0.0

        # Estrai solo la parte dopo </think> dalla risposta generata
        match_format = POST_THINK_PATTERN.search(response)
        if not match_format:
            # Il modello non ha usato il tag </think>: nessun reward di accuratezza
            rewards.append(score)
            continue

        analysis_content = match_format.group(1).strip()

        # --- Confronto numero dello step ---
        gen_step_match = STEP_NUM_PATTERN.search(analysis_content)
        true_step_match = STEP_NUM_PATTERN.search(true_answer)
        if gen_step_match and true_step_match and gen_step_match.group(1) == true_step_match.group(1):
            score += 0.2

        # --- Confronto del comando ---
        gen_command_match = COMMAND_PATTERN.search(analysis_content)
        true_command_match = COMMAND_PATTERN.search(true_answer)

        gen_command = gen_command_match.group(1).strip() if gen_command_match else ""
        true_command = true_command_match.group(1).strip() if true_command_match else ""

        # Se il ground truth non ha un comando, salta il confronto
        if not true_command:
            rewards.append(score)
            continue

        if gen_command == true_command:
            # Comando identico: reward piena
            score += 1.0
        elif gen_command:
            # Reward parziale basata sulla similarità Jaccard tra le parole del comando.
            # Es: "nmap -sV 192.168.1.1" vs "nmap -sV 10.0.0.1" → similarità alta ma non 1.0
            gen_words = set(gen_command.split())
            true_words = set(true_command.split())
            intersection = gen_words & true_words

            union_size = len(gen_words) + len(true_words) - len(intersection)
            if union_size > 0:
                similarity = len(intersection) / union_size
                if similarity > 0.5:
                    score += similarity * 0.7

        rewards.append(score)
    return rewards

def format_reward(completions, **kwargs):
    """
    Reward function per il FORMATO della risposta generata.
    Valuta se il modello ha rispettato la struttura attesa, indipendentemente
    dalla correttezza del contenuto.

    Punteggio massimo: 0.6
      +0.3  se la risposta contiene un blocco <think>...</think> non vuoto
             (il modello ha ragionato prima di rispondere)
      +0.3  se la risposta contiene il formato "=== Step N === Thought: ... Command:"
             (il modello ha rispettato la struttura richiesta)

    Riceve una lista di completions, dove ogni elemento è una lista di messaggi
    del modello (tipicamente [{"role": "assistant", "content": "..."}]).
    """
    # Cattura il contenuto tra <think>...</think>: il ragionamento interno del modello (chain-of-thought).
    THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    # Verifica che la risposta contenga un blocco nel formato:
    #   === Step N ===
    #   Thought: ...
    #   Command: ...
    STEP_PATTERN = re.compile(r"===\s*Step\s*\d+\s*===\s*Thought:.*?Command:", re.DOTALL | re.IGNORECASE)

    rewards = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0

        # Verifica presenza di ragionamento interno (chain-of-thought)
        think_match = THINK_PATTERN.search(response)
        if think_match and think_match.group(1).strip():
            score += 0.3

        # Verifica presenza del formato step/thought/command
        if STEP_PATTERN.search(response):
            score += 0.3

        rewards.append(score)
    return rewards
