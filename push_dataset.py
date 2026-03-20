import os
from huggingface_hub import login, upload_file

# Esegui il login se non l'hai fatto (serve il token con permessi 'write')
login(token=os.environ["HF_TOKEN"])

# Carica solo il file desiderato
upload_file(
    path_or_fileobj="CyberFineWeb.parquet",
    path_in_repo="CyberFineWeb.parquet",
    repo_id="mdomina/CyberFineWeb",
    repo_type="dataset"
)