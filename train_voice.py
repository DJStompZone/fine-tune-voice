"""
Attempting to train a NN on my own voice.
What could *possibly* go wrong?

Copyright
---------
DJ Stomp 2025

License
-------
* Code for training and auto annotation:
    MIT
    See the LICENSE file for full details.

* Weights, fine-tunings, artifacts, etc:
    CC-BY-NC-SA 4.0
    Insofar as is required by the terms stipulated in the hereditary Creative Commons license.
    See the LICENSE file for full details.

* Note:
    For the purposes of disambiguation, any textual material in this project shall be considered to be licensed as MIT unless specifically declared otherwise.
"""

!pip install torch torchvision torchaudio transformers datasets huggingface_hub soundfile tqdm requests python-dotenv
!git clone https://github.com/fishaudio/fish-speech.git
%cd fish-speech
!pip install -e .[stable]
!apt-get install -y sox ffmpeg

!huggingface-cli login

import os
import time
import requests
from dotenv import load_dotenv
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder
from tqdm import tqdm
from datasets import load_dataset

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
HF_REPO_NAME = os.getenv("HF_REPO_NAME")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 20))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", 500))
SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", 3))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", 100))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Because I don't already have enough Discord pings
def send_discord_message(message):
    """
    Sends a message to Discord using the provided webhook URL.
    """
    payload = {"content": f"[**StompNET**] {message}"}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print(f"Discord message sent: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Discord message: {e}")

# Trust nobody, not even oneself... Especially oneself. 
# 😐🔫😐 https://i.imgur.com/Du9vtBy.jpeg
def validate_dataset(dataset_path):
    """
    Validates the dataset structure and content.
    Raises descriptive exceptions if issues are found.
    """
    audio_dir = os.path.join(dataset_path, "audio")
    transcriptions_dir = os.path.join(dataset_path, "transcriptions")

    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio directory is missing: {audio_dir}")
    if not os.path.exists(transcriptions_dir):
        raise FileNotFoundError(f"Transcriptions directory is missing: {transcriptions_dir}")

    audio_files = os.listdir(audio_dir)
    transcription_files = os.listdir(transcriptions_dir)

    missing_transcriptions = []
    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]
        transcription_file = f"{base_name}.txt"
        if transcription_file not in transcription_files:
            missing_transcriptions.append(audio_file)

    if missing_transcriptions:
        raise ValueError(
            f"The following audio files are missing transcriptions: {', '.join(missing_transcriptions)}"
        )

    print(f"Dataset validation complete: {len(audio_files)} audio files and transcriptions found.")
    return len(audio_files)

try:
    send_discord_message("🔍 Validating dataset...")
    num_validated = validate_dataset(DATASET_PATH)
    send_discord_message(f"✅ Dataset validation complete: {num_validated} training records validated.")
except Exception as e:
    send_discord_message(f"⛔ Dataset validation failed: {str(e)}")
    raise

# Lights! 🔦
try:
    audio_model = "fishaudio/fish-speech-1.5"
    send_discord_message(f'🪄 Initializing model and tokenizer: "{audio_model}"...')
    tokenizer = AutoTokenizer.from_pretrained(audio_model)
    model = AutoModelForCausalLM.from_pretrained(audio_model)
    send_discord_message("✅ Model and tokenizer initialized successfully.")
except Exception as e:
    send_discord_message(f"⛔ Model initialization failed: {str(e)}")
    raise

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    logging_dir="./logs",
    logging_steps=LOGGING_STEPS,
    report_to="hub",
    hub_model_id=HF_REPO_NAME,
    hub_token=HUGGINGFACE_TOKEN,
    push_to_hub=True,
    resume_from_checkpoint=True
)

# Camera! 🎥 
try:
    send_discord_message(f"🧮 Loading dataset from {DATASET_PATH}/train.json...")
    train_dataset = load_dataset("json", data_files={"train": f"{DATASET_PATH}/train.json"})["train"]

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    send_discord_message("✅ Dataset loaded and tokenized successfully.")
except Exception as e:
    send_discord_message(f"⛔ Dataset loading or tokenization failed: {str(e)}")
    raise

# Do you even lift, bro? 💪
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

start_time = elapsed_time = time.time()

# Action! 🎬
try:
    send_discord_message("🎯 Starting fine-tuning...")
    send_discord_message("\n".join([
        f"Initiating training session:", 
        f"  {NUM_EPOCHS=}",
        f"  {SAVE_STEPS=}",
        f"  {OUTPUT_DIR=}",
        f"  {BATCH_SIZE=}"
    ]))
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    send_discord_message(f"✅ Fine-tuning completed successfully in {elapsed_minutes:.2f} minutes.")
except Exception as e:
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    send_discord_message(f"⛔ Fine-tuning failed after {elapsed_minutes:.2f} minutes: {str(e)}")
    raise

# Fuck it! Ship it! 🚀
try:
    send_discord_message("🚀 Pushing final model to Hugging Face Hub...")
    trainer.push_to_hub(commit_message="Fine-tuned Fish-Speech model.")
    send_discord_message("✅ Final model pushed to Hugging Face Hub.")
except Exception as e:
    send_discord_message(f"⛔ Failed to push model to Hugging Face Hub: {str(e)}")
    raise