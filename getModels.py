from huggingface_hub import snapshot_download
import os

os.makedirs("meta-llama", exist_ok=True)

snapshot_download(
    repo_id="meta-llama/Llama-2-7b",
    revision="main",
    cache_dir="./meta-llama/7b",
    resume_download=True
)



snapshot_download(
    repo_id="meta-llama/Llama-2-70b-chat-hf",
    revision="main",
    cache_dir="./meta-llama/70b-chat",
    resume_download=True
)
