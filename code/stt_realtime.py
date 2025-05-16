import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import keyboard
import time
import sys
import os

# 설정
samplerate = 16000
recording = []
is_recording = False
stream = None

def audio_callback(indata, frames, time, status):
    if is_recording:
        recording.append(indata.copy())

def start_recording():
    global is_recording, stream, recording
    print("\n[녹음 시작]")
    is_recording = True
    recording = []
    stream = sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback)
    stream.start()

def stop_recording():
    global is_recording, stream
    print("[녹음 종료 및 변환 시작]")
    is_recording = False
    stream.stop()
    stream.close()
    process_audio()

def process_audio():
    try:
        # 녹음된 데이터 합치기
        audio_data = np.concatenate(recording, axis=0)
        wav_path = "recorded.wav"
        wav.write(wav_path, samplerate, (audio_data * 32767).astype(np.int16))

        # Whisper 모델 로드 및 변환
        print("[Whisper 모델 로드 중...]")
        model = whisper.load_model("small")

        print("[음성 인식 중...]")
        result = model.transcribe(wav_path, language="ko")

        recognized_text = result.get("text", "").strip()

        if not recognized_text:
            print("[경고] 인식된 텍스트가 없습니다. 발화를 확인해주세요.")
        else:
            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(recognized_text)
            print("[output.txt에 텍스트 저장 완료]")

            print("\n[output.txt의 내용]")
            print(recognized_text)

    except Exception as e:
        print(f"[오류 발생] {e}")

    print("\n[프로그램 종료]")
    sys.exit()

def main():
    print("R 키 → 녹음 시작 / X 키 → 녹음 종료 및 텍스트 변환")
    
    # R 키가 눌릴 때까지 대기
    while True:
        if keyboard.is_pressed("r"):
            start_recording()
            while keyboard.is_pressed("r"):
                time.sleep(0.1)
            break

    # X 키가 눌릴 때까지 대기
    while True:
        if keyboard.is_pressed("x"):
            stop_recording()
            break
        time.sleep(0.1)

if __name__ == "__main__":
    main()
