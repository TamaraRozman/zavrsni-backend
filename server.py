import os
import uuid
import torchaudio
from fastapi import FastAPI, UploadFile, File
from speechbrain.inference.separation import SepformerSeparation
from faster_whisper import WhisperModel

# ===== CONFIG =====
LANGUAGE = "hr"
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
SEPFORMER_SR = 8000

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

sep_model = None
whisper_model = None


# =========================
# LOAD MODELS ON STARTUP
# =========================
@app.on_event("startup")
async def load_models():
    global sep_model, whisper_model

    print("Loading Sepformer model...")
    sep_model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="pretrained_models/sepformer"
    )

    print("Loading Whisper model...")
    whisper_model = WhisperModel("small", compute_type="int8")

    print("Models loaded successfully.")


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "running"}


# =========================
# TRANSCRIBE ENDPOINT
# =========================
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    if sep_model is None or whisper_model is None:
        return {"error": "Models not loaded yet"}

    file_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    print("Separating audio sources...")
    est_sources = sep_model.separate_file(path=input_path)
    num_sources = est_sources.shape[2]

    all_segments = []

    for i in range(num_sources):
        speaker_label = f"Govornik {i+1}"
        speaker_path = os.path.join(TEMP_DIR, f"{file_id}_spk{i}.wav")

        waveform = est_sources[:, :, i].detach().cpu()
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val * 0.9

        waveform_resampled = torchaudio.transforms.Resample(
            orig_freq=SEPFORMER_SR,
            new_freq=16000
        )(waveform)

        torchaudio.save(speaker_path, waveform_resampled, 16000)

        print(f"Transcribing {speaker_label}...")
        segments, _ = whisper_model.transcribe(
            speaker_path,
            language=LANGUAGE
        )

        for segment in segments:
            all_segments.append({
                "speaker": speaker_label,
                "start": segment.start,
                "transcript": segment.text.strip()
            })

        os.remove(speaker_path)

    os.remove(input_path)

    # Sort by time
    all_segments.sort(key=lambda x: x["start"])

    # Remove timestamps before returning
    final_output = [
        {
            "speaker": seg["speaker"],
            "transcript": seg["transcript"]
        }
        for seg in all_segments
    ]

    return final_output
