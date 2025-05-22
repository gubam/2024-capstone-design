from .inference import run_sign_inference
from .gpt_api import generate_sentence_with_gpt,generate_gloss_with_gpt
from .tts_stt_output import save_tts_output, stt_output

#person = 의사 or 환자
# 비디오 소스 들어오면 센텐스로 변화
def sign_to_audio(video_source):
    words = run_sign_inference(video_source)#비디오,pt 파일명
    sentence = generate_sentence_with_gpt(words, "환자")
    save_tts_output(sentence)
    return sentence

def audio_to_sign():
    sentence = stt_output()
    gloss = generate_gloss_with_gpt(sentence)
    result = gloss.split(",")
    return result

# print(audio_to_sign())
# result = generate_gloss_with_gpt("안녕하세요 머리가 아프네요")
# result = result.split(",")
# print(result)
# result = sign_to_audio("C:/Users/82109/Desktop/테스트셋-20250522T040652Z-1-001/테스트셋/감사합니다/20250521_181636.mp4")
