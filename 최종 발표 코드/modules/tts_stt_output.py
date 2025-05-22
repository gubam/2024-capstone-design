from gtts import gTTS
import os
import whisper

#tts 출력 함수
def save_tts_output(sen):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "file", "tts_output.mp3")
    tts = gTTS(text=sen, lang='ko')
    tts.save(MODEL_PATH)

#오디오 인풋 들어오면 문장 출력
#출력은 그냥 문자열 형태
def stt_output():
    model = whisper.load_model("small")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(BASE_DIR, "..", "file", "stt_input.mp3")
    result = model.transcribe(audio_file)
    return result["text"]
