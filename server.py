import whisper

# Load Whisper model (choose 'base', 'small', 'medium', or 'large' for better accuracy)
whisper_model = whisper.load_model("small")

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    audio_path = "temp_audio.wav"
    file.save(audio_path)

    # Separate speakers
    est_sources = sep_model.separate_file(path=audio_path)
    num_speakers = est_sources.shape[2]
    sample_rate = 8000

    results = []
    
    for i in range(num_speakers):
        speaker_audio_path = f"speaker_{i+1}.wav"
        torchaudio.save(speaker_audio_path, est_sources[:, :, i].detach().cpu(), sample_rate)

        # Transcribe using Whisper (Croatian language supported)
        transcription = whisper_model.transcribe(speaker_audio_path, language="hr")["text"]

        results.append({
            "speaker": f"Speaker {i+1}",
            "transcription": transcription
        })

    return jsonify({"speakers": results})
