from fastapi import FastAPI, UploadFile, File
import os
import torchaudio
from speechbrain.inference.separation import SepformerSeparation
from faster_whisper import WhisperModel
import uuid

app = FastAPI()

# Load models once at startup
sep_model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)

# Use small if CPU is slow
model = WhisperModel("medium", compute_type="int8")

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    
    # Save uploaded file
    input_path = f"temp_{uuid.uuid4()}.wav"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Separate sources
    est_sources = sep_model.separate_file(path=input_path)

    # Remove batch dimension
    est_sources = est_sources.squeeze(0)  # shape: [time, sources]

    os.makedirs("output", exist_ok=True)

    results = []
    num_sources = est_sources.shape[1]

    for i in range(num_sources):
        filename = f"source{i+1}_{uuid.uuid4()}.wav"
        filepath = os.path.join("output", filename)

        # Extract single speaker
        source_audio = est_sources[:, i].detach().cpu().unsqueeze(0)

        # Save at original 8000 Hz (NO resampling)
        torchaudio.save(
            filepath,
            source_audio,
            8000,
            encoding="PCM_S",
            bits_per_sample=16
        )

        # Transcribe (faster-whisper handles resampling internally)
        transcription = ""
        segments, _ = model.transcribe(filepath, language="hr")

        for segment in segments:
            transcription += segment.text

        results.append({
            "speaker": f"Govornik {i+1}",
            "transcript": transcription.strip()
        })

    # Cleanup input file
    os.remove(input_path)

    return results
