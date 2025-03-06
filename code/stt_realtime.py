'''
필요한 패키지 설치
pip install openai-whisper 
pip install pyaudio (실시간 음성 인식) 
pip install numpy (오디오 처리)
'''

import whisper
import pyaudio
import numpy as np
import wave
import tempfile
import os

model = whisper.load_model("base")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

try:
    while True:
        frames = []
        print("Listening...")

        for _ in range(0, int(RATE / CHUNK * 5)):  
            data = stream.read(CHUNK)
            frames.append(data)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav.close()

        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        result = model.transcribe(temp_wav.name, language="ko")
        print("Recognized:", result["text"])

        os.remove(temp_wav.name)

except KeyboardInterrupt:
    print("Stopped.")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
