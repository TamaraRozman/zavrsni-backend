from flask import Flask, request, jsonify
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import io

app = Flask(__name__)

# Load SpeechBrain model
model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="tmpdir")

@app.route('/separate', methods=['POST'])
def separate_audio():
    file = request.files['audio']
    audio_bytes = file.read()

    # Convert bytes to tensor
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

    # Separate voices
    separated = model.separate_batch(waveform)

    # Save each speaker's audio
    speaker_files = {}
    for i, speaker_audio in enumerate(separated):
        speaker_path = f"speaker_{i}.wav"
        torchaudio.save(speaker_path, speaker_audio.unsqueeze(0), 8000)
        speaker_files[f"Speaker {i}"] = speaker_path

    return jsonify({"speakers": list(speaker_files.values())})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
