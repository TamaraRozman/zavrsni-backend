import asyncio
import websockets
import base64
import io
import torch
from speechbrain.inference import WhisperASR

# Load Whisper ASR model (supports Croatian)
asr_model = WhisperASR.from_hparams(source="speechbrain/whisper-large-v2", savedir="whisper_model")

async def process_audio(websocket, path):
    print("Client connected")
    
    audio_chunks = bytearray()
    
    async for message in websocket:
        audio_data = base64.b64decode(message)  # Decode Base64
        audio_chunks.extend(audio_data)

        if len(audio_chunks) > 16000 * 5:  # Process every 5 seconds
            print("Processing Croatian audio...")
            
            # Convert to tensor
            audio_tensor = torch.tensor(list(audio_chunks), dtype=torch.float32).unsqueeze(0)
            
            # Transcribe in Croatian
            transcript = asr_model.transcribe_batch(audio_tensor, language="hr")
            
            await websocket.send(transcript)  # Send transcript back to Flutter
            audio_chunks.clear()

async def main():
    async with websockets.serve(process_audio, "0.0.0.0", 8765):  
        await asyncio.Future()  # Run server forever

if __name__ == "__main__":
    asyncio.run(main())
