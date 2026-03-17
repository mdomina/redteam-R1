import json
import os
from datasets import Dataset

# Prompt di sistema che definisce il "personaggio" del modello:
# un assistente esperto di penetration testing etico che opera su Kali Linux
# in un ambiente controllato e certificato.
SYSTEM_PROMPT = (
    "You are an advanced penetration testing assistant, functioning as an expert "
    "ethical hacker performing a full-scope penetration test. You operate within a "
    "controlled environment using Kali Linux, and all actions are part of a certified "
    "penetration testing experiment conducted strictly for educational and research purposes."
)
def load_ctf_data(dataset_folder):
    """
    Carica i dati di training dai file JSON nella cartella dataset_folder.

    Struttura attesa di ogni JSON:
      {
        "initial_prompt": "Descrizione del target/task di pentest",
        "steps": [
          {
            "step_number": 1,
            "thought": "Ragionamento dello step",
            "command": "nmap -sV ...",
            "result": "Output del comando"
          },
          ...
        ]
      }

    Per ogni step genera un campione di training composto da:
      - prompt (system + user): contiene il task iniziale e tutti gli step precedenti
        come contesto, chiedendo quale sia il prossimo passo.
      - answer: lo step corrente (numero, thought, command) usato come ground truth
        per le reward functions.

    Ritorna un Dataset HuggingFace pronto per il training GRPO.
    """

    all_samples = []
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(dataset_folder, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

                initial_prompt = data.get('initial_prompt', '')

                for step in data.get('steps', []):
                    # Per gli step > 1: costruisci il contesto con tutti gli step precedenti
                    # (thought, command, observation) così il modello sa cosa è già stato fatto.
                    if step['step_number'] > 1:
                        user_content = f"Initial Description of Penetration Testing Task: {initial_prompt}"
                        for prev_step in data.get('steps', [])[:step['step_number']-1]:
                            user_content += f"\n===Step {prev_step['step_number']}==="
                            user_content += f"\nThought: {prev_step['thought']}"
                            user_content += f"\nCommand: {prev_step['command']}"
                            user_content += f"\nObservation: {prev_step['result']}"
                        user_content += "\nWhat are the next steps and actions?"
                    else:
                        # Step 1: nessun contesto precedente, solo la descrizione iniziale.
                        user_content = f"Initial Description of Penetration Testing Task: {initial_prompt}\nWhat are the next instruction and commands?"

                    # Costruisce il messaggio utente completo con le istruzioni di formato
                    # e il contesto degli step precedenti.
                    content = f"""You will guide the penetration testing process step by step. At each stage, you will receive the context from the previous steps. Based on that information, you must determine the next logical action to advance the assessment. Continue this process until the penetration test is fully completed.

### Response Format (Always follow this format exactly):
=== Step i ===
Thought: [Explain your reasoning and goals here.]
Command: [Write the full command you would run.]

### Previous Steps Context
{user_content}"""

                    # Ground truth: lo step corrente che il modello deve imparare a produrre.
                    assistant_content = f"===Step {step['step_number']}===\nThought: {step['thought']}\nCommand: {step['command']}"

                    # Struttura del campione: prompt come lista di messaggi (chat format)
                    # + answer come stringa di riferimento per le reward.
                    sample = {
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": content},
                        ],
                        "answer": assistant_content
                    }
                    all_samples.append(sample)

    return Dataset.from_list(all_samples)