from flask import Flask
from flask_socketio import SocketIO
import torchaudio
import io
import torch
from speechbrain.inference import SepformerSeparation as separator
from speechbrain.inference import EncoderDecoderASR

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSockets for real-time streaming

# Load models
separation_model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="tmpdir")
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en", savedir="asr_model")

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handles incoming audio chunks from Flutter"""

    # Convert bytes to tensor
    audio_data = io.BytesIO(data)
    waveform, sample_rate = torchaudio.load(audio_data)

    # Separate speakers
    separated_waveforms = separation_model.separate_batch(waveform)

    transcripts = {}
    for i, speaker_waveform in enumerate(separated_waveforms):
        # Transcribe the separated voice
        transcript = asr_model.transcribe_batch(speaker_waveform.unsqueeze(0))
        transcripts[f"speaker_{i}"] = transcript

    # Send transcriptions back to Flutter
    socketio.emit('transcription', transcripts)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
