from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator
import whisper
import uuid

app = FastAPI()

sep_model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')
asr_model = whisper.load_model("tiny")

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    input_path = f"temp_{uuid.uuid4()}.wav"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    est_sources = sep_model.separate_file(path=input_path)
    os.makedirs("output", exist_ok=True)

    results = []
    num_sources = est_sources.shape[2]

    for i in range(num_sources):
        filename = f"source{i+1}_{uuid.uuid4()}.wav"
        filepath = os.path.join("output", filename)
        torchaudio.save(filepath, est_sources[:, :, i].detach().cpu(), 8000)

        transcription = asr_model.transcribe(filepath, language="hr")["text"]

        results.append({
            "speaker": f"Govornik {i + 1}",
            "transcript": transcription
        })

    print(results)
    return {"results": results}
