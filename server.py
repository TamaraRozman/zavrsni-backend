import os
import uuid
from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

LANGUAGE = "hr"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

ACCESS_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

print("Loading diarization model...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=ACCESS_TOKEN
)

print("Loading Whisper...")
whisper = WhisperModel("medium", compute_type="int8")

print("Models loaded.")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    diarization = pipeline(input_path)
    segments, _ = whisper.transcribe(input_path, language=LANGUAGE)

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
            "text": seg.text.strip()
        })

    os.remove(input_path)
    return labeled
