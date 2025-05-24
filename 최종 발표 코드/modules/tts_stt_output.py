from gtts import gTTS
import os
import whisper

#tts 출력 함수
def save_tts_output(sen):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "file", "tts_output.mp3")
        # 파일이 열려 있을 가능성이 있으므로 안전하게 삭제 시도
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
    except PermissionError:
        print("❌ MP3 파일이 사용 중입니다. 재생 중이거나 닫히지 않았습니다.")
        return

    try:
        tts = gTTS(text=sen, lang='ko')
        tts.save(MODEL_PATH)
    except Exception as e:
        print(f"❌ TTS 저장 실패: {e}")
    # tts = gTTS(text=sen, lang='ko')
    # tts.save(MODEL_PATH)

#오디오 인풋 들어오면 문장 출력
#출력은 그냥 문자열 형태
def stt_output():
    model = whisper.load_model("small")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(BASE_DIR, "..", "file", "stt_input.mp3")
    result = model.transcribe(audio_file)
    return result["text"]
