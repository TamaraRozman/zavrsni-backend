from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
import uuid

app = FastAPI()
model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

BASE_URL = "https://web-production-80392.up.railway.app"
@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    # Save input
    input_path = f"temp_{uuid.uuid4()}.wav"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Run separation
    est_sources = model.separate_file(path=input_path)
    os.makedirs("output", exist_ok=True)

    # Save all separated sources dynamically
    saved_files = []
    num_sources = est_sources.shape[2]

    for i in range(num_sources):
        filename = f"source{i+1}_{uuid.uuid4()}.wav"
        filepath = os.path.join("output", filename)
        torchaudio.save(filepath, est_sources[:, :, i].detach().cpu(), 8000)
        saved_files.append(f"{BASE_URL}/download/{filename}")

    # Clean up input
    os.remove(input_path)

    # Return paths of all saved files
    return {"sources": saved_files}

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join("output", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='audio/wav', filename=filename)
    return {"error": "File not found"}
