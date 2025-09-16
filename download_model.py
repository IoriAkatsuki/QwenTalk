
import os
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- Configuration ---
MODEL_ID = "Qwen/Qwen1.5-7B-Chat-GGUF" # Example model, replace if needed
MODEL_FILENAME = "qwen1_5-7b-chat-q5_k_m.gguf" # Example file, replace if needed
MODEL_DIR = "Qwen3-8B-nf4-ov"

def download_model_with_progress(repo_id, filename, local_dir):
    """Downloads a file from Hugging Face Hub with a progress bar."""
    if not os.path.exists(local_dir):
        print(f"Creating directory: {local_dir}")
        os.makedirs(local_dir)

    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        print(f"Model file already exists: {local_path}")
        return

    print(f"Downloading {filename} from {repo_id}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"Successfully downloaded model to {local_path}")
    except Exception as e:
        print(f"An error occurred during download: {e}")
        # Clean up partially downloaded file if something went wrong
        if os.path.exists(local_path + ".incomplete"):
            os.remove(local_path + ".incomplete")

if __name__ == "__main__":
    # Ensure the script is run from the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("--- Starting Model Download ---")
    download_model_with_progress(repo_id=MODEL_ID, filename=MODEL_FILENAME, local_dir=MODEL_DIR)
    print("--- Model Download Finished ---")
