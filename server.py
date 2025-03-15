from flask import Flask, request, jsonify
import torch
from speechbrain.pretrained import Tacotron2, HIFIGAN

app = Flask(__name__)

# Load the SpeechBrain separator model
separator = torch.hub.load("speechbrain/speechbrain", "sepformer-wsj02mix")

@app.route("/separate", methods=["POST"])
def separate():
    audio_file = request.files["file"]
    audio_path = "uploaded_audio.wav"
    audio_file.save(audio_path)
    
    # Use SpeechBrain to separate voices in the audio
    separated_audio = separator.separate_file(audio_path)

    # Save the separated audio to send back to Flutter
    output_path = "separated_audio.wav"
    separated_audio.save(output_path)
    
    return jsonify({"message": "Separation successful", "audio_url": output_path})

if __name__ == "__main__":
    app.run(debug=True)
