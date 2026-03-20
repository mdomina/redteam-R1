import json
import os
import anthropic
from datasets import Dataset
from tqdm import tqdm
from utils.common import SYSTEM_PROMPT


_COMPRESS_PROMPT = """\
Summarize this penetration testing observation in max 1 sentence.
Keep: discovered ports/services, credentials, hashes, file paths, error messages, key findings.
Remove: verbose explanations and filler text.
Return ONLY the summary.

Observation: {observation}"""


def _compress_observation(client: anthropic.Anthropic, observation: str, cache: dict) -> str:
    if observation in cache:
        return cache[observation]
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": _COMPRESS_PROMPT.format(observation=observation)}],
    )
    result = msg.content[0].text.strip()
    cache[observation] = result
    return result


def _load_checkpoint(checkpoint_path: str) -> dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {"completed_files": [], "obs_cache": {}}


def _save_checkpoint(checkpoint_path: str, completed_files: list, obs_cache: dict):
    with open(checkpoint_path, "w") as f:
        json.dump({"completed_files": completed_files, "obs_cache": obs_cache}, f)


def load_ctf_data_compressed(dataset_folder: str, output_dir: str = "dataset/ctf_compressed_hf") -> Dataset:
    """
    Come load_ctf_data, ma nel contesto degli step precedenti (step_number > 1)
    comprime le Observation con un LLM (claude-haiku) per ridurre la lunghezza
    del prompt mantenendo le informazioni chiave per il GRPO.

    Richiede ANTHROPIC_API_KEY nell'ambiente.
    """
    client = anthropic.Anthropic()  # legge ANTHROPIC_API_KEY dall'environment
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = _load_checkpoint(checkpoint_path)
    completed_files: list = checkpoint["completed_files"]
    obs_cache: dict = checkpoint["obs_cache"]

    all_samples = []
    files = [f for f in os.listdir(dataset_folder) if f.endswith('.json')]

    for filename in tqdm(files, desc="Files", unit="file"):
        if filename in completed_files:
            continue
        filepath = os.path.join(dataset_folder, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        initial_prompt = data.get('initial_prompt', '')
        steps = data.get('steps', [])

        for step in tqdm(steps, desc=filename, unit="step", leave=False):
            if step['step_number'] > 1:
                user_content = f"Initial Description of Penetration Testing Task: {initial_prompt}"
                for prev_step in steps[:step['step_number'] - 1]:
                    compressed_obs = _compress_observation(client, prev_step['result'], obs_cache)
                    user_content += f"\n===Step {prev_step['step_number']}==="
                    user_content += f"\nThought: {prev_step['thought']}"
                    user_content += f"\nCommand: {prev_step['command']}"
                    user_content += f"\nObservation: {compressed_obs}"
                user_content += "\nWhat are the next steps and actions?"
            else:
                user_content = f"Initial Description of Penetration Testing Task: {initial_prompt}\nWhat are the next instruction and commands?"

            content = f"""You will guide the penetration testing process step by step. At each stage, you will receive the context from the previous steps. Based on that information, you must determine the next logical action to advance the assessment. Continue this process until the penetration test is fully completed.

### Response Format (Always follow this format exactly):
=== Step i ===
Thought: [Explain your reasoning and goals here.]
Command: [Write the full command you would run.]

### Previous Steps Context
{user_content}"""

            assistant_content = f"===Step {step['step_number']}===\nThought: {step['thought']}\nCommand: {step['command']}"

            sample = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "answer": assistant_content
            }
            all_samples.append(sample)

        completed_files.append(filename)
        _save_checkpoint(checkpoint_path, completed_files, obs_cache)

    dataset = Dataset.from_list(all_samples)
    dataset = dataset.map(lambda x: {"prompt_length": sum(len(m["content"]) for m in x["prompt"])})
    dataset = dataset.sort("prompt_length")
    dataset = dataset.remove_columns(["prompt_length"])
    dataset.save_to_disk(output_dir)
    print(f"Dataset salvato in: {output_dir}")
    return dataset


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "dataset/ctf"
    load_ctf_data_compressed(folder)
