import os
import uuid
import subprocess
import torch
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment

# -------- CONFIG --------
LANGUAGE = "hr"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

ACCESS_TOKEN = os.getenv("HF_TOKEN")

# Limit thread explosion on Railway
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

app = FastAPI()

# -------- LOAD MODELS (ONCE) --------
print("Loading diarization model (fast CPU version)...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=ACCESS_TOKEN
)
pipeline.to(torch.device("cpu"))

print("Loading Whisper (CPU optimized)...")
whisper = WhisperModel(
    "base",              # MUCH faster than small
    device="cpu",
    compute_type="int8"  # important for CPU speed
)

print("Models loaded.")


# -------- FAST AUDIO PREPROCESS --------
def preprocess_audio(input_path, output_path):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",        # mono
            "-ar", "16000",    # 16kHz
            output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())

    raw_path = os.path.join(TEMP_DIR, f"{file_id}_raw.wav")
    processed_path = os.path.join(TEMP_DIR, f"{file_id}_16k.wav")

    # Save uploaded file
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    # Downsample for speed
    preprocess_audio(raw_path, processed_path)
    os.remove(raw_path)

    # ----- DIARIZATION -----
    diarization = pipeline(processed_path)

    # ----- TRANSCRIPTION -----
    segments, _ = whisper.transcribe(
        processed_path,
        language=LANGUAGE,
        beam_size=1  # IMPORTANT: faster decoding
    )

    labeled = []

    # Convert diarization to list ONCE (avoid nested heavy calls)
    diar_segments = list(diarization.itertracks(yield_label=True))

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
            "transcript": seg.text.strip()
        })

    os.remove(processed_path)

    return labeled
