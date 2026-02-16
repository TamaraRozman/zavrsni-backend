import os
import uuid
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from fastapi.middleware.cors import CORSMiddleware

# === CONFIG ===
LANGUAGE = "hr"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

ACCESS_TOKEN = os.getenv("HF_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set!")

app = FastAPI(title="Audio Transcription & Diarization")

# === CORS (allow local mobile app to access server) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # za razvoj lokalno
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === LOAD MODELS AT STARTUP ===
print("⏳ Loading Pyannote diarization model...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=ACCESS_TOKEN
)
print("✅ Diarization model loaded.")

print("⏳ Loading Whisper model...")
whisper = WhisperModel("medium", compute_type="int8")
print("✅ Whisper model loaded.")

# === ENDPOINT ===
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    try:
        # Save uploaded audio
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Run diarization
        diarization = pipeline(input_path)

        # Run transcription
        segments, _ = whisper.transcribe(input_path, language=LANGUAGE)

        # Align speakers with transcription
        labeled = []
        for seg in segments:
            t0, t1 = seg.start, seg.end
            max_overlap = 0.0
            best_speaker = "Unknown"

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(t0, turn.start)
                overlap_end = min(t1, turn.end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker

            labeled.append({
                "speaker": best_speaker.replace("SPEAKER_", "Govornik "),
                "transcript": seg.text.strip()
            })

        return labeled

    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
