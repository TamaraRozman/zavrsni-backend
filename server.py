from flask import Flask
from flask_socketio import SocketIO, emit
import torchaudio
from speechbrain.pretrained import SepformerSeparation
from faster_whisper import Whisper
import tempfile
import os

# Initialize Flask and WebSockets
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load models
separation_model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="tmp")
whisper_model = Whisper("medium")

# Process received audio
def process_audio(audio_path):
    waveform, _ = torchaudio.load(audio_path)
    separated = separation_model.separate_batch(waveform)

    results = []
    for i, speaker_audio in enumerate(separated):
        temp_file = f"temp_speaker_{i}.wav"
        torchaudio.save(temp_file, speaker_audio, 16000)

        segments, _ = whisper_model.transcribe(temp_file)
        text = " ".join([seg.text for seg in segments])
        results.append({"speaker": f"Speaker {i+1}", "text": text})
        
        os.remove(temp_file)

    return results

# WebSocket connection
@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(data)
        temp_audio.close()

        transcriptions = process_audio(temp_audio.name)
        os.remove(temp_audio.name)

        emit("transcription", transcriptions)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
