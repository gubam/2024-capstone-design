'''
필요한 패키지 설치
pip install openai-whisper 
pip install pyaudio (실시간 음성 인식) 
'''

import whisper

# Whisper 모델 불러오기 (small 모델 사용)
model = whisper.load_model("small")

# 변환할 오디오 파일 경로
audio_file = "audio.mp3" 

# 오디오 파일을 텍스트로 변환
print(" 음성 인식 중...")
result = model.transcribe(audio_file)

# 변환된 텍스트 출력
print("\n 변환된 텍스트:")
print(result["text"])

# 변환된 텍스트를 파일로 저장
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("\n 변환 완료! 'output.txt'에 저장되었습니다.")
