from modules.inference import run_sign_inference
from modules.gpt_api import generate_sentence_with_gpt

#person = 의사 or 환자
# 비디오 소스 들어오면 센텐스로 변화
def sign_to_sentence(video_source):
    words = run_sign_inference(video_source)
    sentence = generate_sentence_with_gpt(words, "환자")
    return sentence

sign_to_sentence("C:/Users/82109/Desktop/test1/sen/나아프다.mp4")