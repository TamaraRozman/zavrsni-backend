from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
import uuid

app = FastAPI()
model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    # Save input
    input_path = f"temp_{uuid.uuid4()}.wav"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Run separation
    est_sources = model.separate_file(path=input_path)
    os.makedirs("output", exist_ok=True)
    source1_path = f"output/source1_{uuid.uuid4()}.wav"
    source2_path = f"output/source2_{uuid.uuid4()}.wav"
    torchaudio.save(source1_path, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(source2_path, est_sources[:, :, 1].detach().cpu(), 8000)

    # Clean up
    os.remove(input_path)

    # Return paths (could also return as base64)
    return {
        "source1": source1_path,
        "source2": source2_path
    }

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join("output", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='audio/wav', filename=filename)
    return {"error": "File not found"}
