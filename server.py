import os
import uuid
import json
import subprocess
import torch
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# ===== CONFIG =====
LANGUAGE = "hr"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")  # set in Railway variables

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

app = FastAPI()

pipeline = None
whisper = None


# ===== LOAD MODELS ON STARTUP =====
@app.on_event("startup")
def load_models():
    global pipeline, whisper

    print("Loading diarization model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    pipeline.to("cpu")

    print("Loading Whisper...")
    whisper = WhisperModel(
        "base",              # use base on Railway, NOT medium
        device="cpu",
        compute_type="int8"
    )

    print("Models loaded successfully.")


# ===== AUDIO PREPROCESS =====
def preprocess(input_path, output_path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ===== ENDPOINT =====
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    raw_path = os.path.join(TEMP_DIR, f"{file_id}_raw.wav")
    processed_path = os.path.join(TEMP_DIR, f"{file_id}_16k.wav")

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    preprocess(raw_path, processed_path)
    os.remove(raw_path)

    diarization = pipeline(processed_path)

    segments, _ = whisper.transcribe(
        processed_path,
        language=LANGUAGE,
        beam_size=1
    )

    diar_segments = list(diarization.itertracks(yield_label=True))
    labeled = []

    for seg in segments:
        t0, t1 = seg.start, seg.end
        best_speaker = "Unknown"
        max_overlap = 0.0

        for turn, _, speaker in diar_segments:
            overlap = max(
                0.0,
                min(t1, turn.end) - max(t0, turn.start)
            )
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        labeled.append({
            "speaker": best_speaker.replace("SPEAKER_", "Govornik "),
            "text": seg.text.strip()
        })

    os.remove(processed_path)

    return labeled
