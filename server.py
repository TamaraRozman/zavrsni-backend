import os
import uuid
import subprocess
import shutil
import torch
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# =============================
# CONFIG
# =============================

LANGUAGE = "hr"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

ACCESS_TOKEN = os.getenv("HF_TOKEN")  # set this in Railway Variables

# Prevent Railway CPU explosion
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# =============================
# FIND FFMPEG SAFELY
# =============================

FFMPEG_PATH = shutil.which("ffmpeg")
if not FFMPEG_PATH:
    raise RuntimeError("ffmpeg not found in container. Make sure nixpacks installs it.")

print(f"Using ffmpeg at: {FFMPEG_PATH}")

# =============================
# APP INIT
# =============================

app = FastAPI()

print("Loading diarization model...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=ACCESS_TOKEN
)
pipeline.to("cpu")

print("Loading Whisper model...")
whisper = WhisperModel(
    "base",          # safer for Railway than medium
    device="cpu",
    compute_type="int8"
)

print("Models loaded successfully.")

# =============================
# AUDIO PREPROCESS
# =============================

def preprocess(input_path, output_path):
    subprocess.run(
        [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

# =============================
# ENDPOINT
# =============================

@app.post("/separate")
async def transcribe(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    raw_path = os.path.join(TEMP_DIR, f"{file_id}_raw.wav")
    processed_path = os.path.join(TEMP_DIR, f"{file_id}_16k.wav")

    # Save uploaded file
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # Convert to 16k mono
    preprocess(raw_path, processed_path)
    os.remove(raw_path)

    # =============================
    # DIARIZATION
    # =============================

    diarization = pipeline(processed_path)
    diar_segments = list(diarization.itertracks(yield_label=True))

    # =============================
    # TRANSCRIPTION
    # =============================

    segments, _ = whisper.transcribe(
        processed_path,
        language=LANGUAGE,
        beam_size=1
    )

    labeled = []

    for seg in segments:
        t0, t1 = seg.start, seg.end
        best_speaker = "Unknown"
        max_overlap = 0.0

        for turn, _, speaker in diar_segments:
            overlap = max(0.0, min(t1, turn.end) - max(t0, turn.start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        labeled.append({
            "speaker": best_speaker.replace("SPEAKER_", "Govornik "),
            "text": seg.text.strip()
        })

    os.remove(processed_path)

    return labeled
