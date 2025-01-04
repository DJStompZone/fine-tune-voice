"""
An attempt to autonomously annotate many hours of raw training data.
What could *possibly* go wrong?
"""

!pip install git+https://github.com/openai/whisper.git
!pip install ffmpeg-python pydub tqdm

import os
import whisper
from tqdm import tqdm
from pydub import AudioSegment
import random
import math

# Friendly reminder 😁
INPUT_AUDIO = "HEY DIPSHIT, YOU FORGOT TO SET THE FILE PATH!"
if INPUT_AUDIO.startswith("H") and not os.path.exists(INPUT_AUDIO):
    print(f"{INPUT_AUDIO} :D")
OUTPUT_DIR = "./transcriptions/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Whisper model (large)...")
model = whisper.load_model("large")

# Fruit ninja, but for data...
# And without the fruit. Or the ninjas.
# ...What was I talking about again?
def split_audio(file_path, min_length_sec=5, max_length_sec=15):
    """
    Splits a long audio file into smaller chunks suitable for TTS fine-tuning.

    Args:
        file_path (str): Path to the input audio file.
        min_length_sec (int): Minimum chunk length in seconds.
        max_length_sec (int): Maximum chunk length in seconds.

    Returns:
        List of chunk file paths.
    """
    print(f"Splitting {file_path} into {min_length_sec}-{max_length_sec} second chunks...")
    audio = AudioSegment.from_file(file_path)
    duration = len(audio) # In milliseconds
    chunk_paths = []
    start_ms = 0

    while start_ms < duration:
        # There's no way using random here will haunt me later :)
        chunk_length = random.randint(min_length_sec * 1000, max_length_sec * 1000)
        # It's fine. It's probably fine.
        end_ms = min(start_ms + chunk_length, duration)
        
        chunk = audio[start_ms:end_ms]
        if len(chunk) >= min_length_sec * 1000:
            chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{start_ms // 1000:05d}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunk_paths.append(chunk_path)
        
        start_ms = end_ms

    print(f"Split into {len(chunk_paths)} chunks.")
    # It's fine.
    return chunk_paths

chunk_paths = split_audio(INPUT_AUDIO)

# Write that down! Write that down!
def transcribe_chunks(chunk_paths, model, output_dir):
    """
    Transcribes audio chunks using OpenAI Whisper.

    Args:
        chunk_paths (list): List of audio chunk file paths.
        model (whisper.Model): Pre-loaded Whisper model.
        output_dir (str): Directory to save transcriptions.
    """
    print("Transcribing audio chunks...")
    for chunk_path in tqdm(chunk_paths):
        result = model.transcribe(chunk_path)
        
        transcription_path = os.path.join(output_dir, os.path.basename(chunk_path).replace(".mp3", ".txt"))
        with open(transcription_path, "w") as f:
            f.write(result["text"])
        
        segments_path = os.path.join(output_dir, os.path.basename(chunk_path).replace(".mp3", "_segments.txt"))
        with open(segments_path, "w") as f:
            for segment in result["segments"]:
                f.write(f"[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text']}\n")
    print("Transcription complete.")

# Mmm. Chunky style.
transcribe_chunks(chunk_paths, model, OUTPUT_DIR)


def combine_transcriptions(output_dir, combined_file="combined_transcription.txt"):
    """
    Combines all transcription files into one.
    
    "The gods forced him to roll an immense boulder up a hill only for it to roll back down every time it neared the top, repeating this action for eternity."

    Args:
        output_dir (str): Directory containing transcription files.
        combined_file (str): Output file for combined transcription.
    """
    print("Combining transcriptions...")
    transcription_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".txt") and "_segments" not in f])
    combined_path = os.path.join(output_dir, combined_file)
    with open(combined_path, "w") as combined:
        for file in transcription_files:
            file_path = os.path.join(output_dir, file)
            with open(file_path, "r") as f:
                combined.write(f.read() + "\n")
    print(f"Combined transcription saved to {combined_path}")

combine_transcriptions(OUTPUT_DIR)