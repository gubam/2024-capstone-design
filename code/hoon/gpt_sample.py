import random
from openai import OpenAI

# ✅ OpenAI API 키 설정
client = OpenAI(api_key=#api key 넣어주세용)

# ✅ 의료 상황용 단어 리스트 (명사 + 동사/형용사)
nouns = [
    '환자', '의사', '배', '머리', '가슴', '다리', '팔', 
    '약', '병원', '기침', '열', '진료', '증상', '통증'
]
verbs = [
    '아프다', '먹다', '처방하다', '진찰하다', '기다리다', 
    '눕다', '걷다', '있다', '없다', '말하다'
]

# ✅ 단어 랜덤 선택 (명사 1~2개 + 동사 2개)
def get_random_medical_words():
    selected_nouns = random.sample(nouns, k=random.randint(1, 2))
    selected_verbs = random.sample(verbs, k=2)
    return selected_nouns + selected_verbs

# ✅ GPT에게 문장 생성 요청
def generate_sentence_with_gpt(words):
    prompt = (
        f"다음 단어들은 수어의 gloss입니다. "
        f"이 단어들만 사용해서 자연스럽고 문법에 맞는 한국어 문장으로 바꿔줘. "
        f"필요하다면 조사나 어미는 붙여도 되지만, 새로운 단어나 동사는 절대 추가하지 마. "
        f"단어들: {', '.join(words)}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()

# ✅ 실행 테스트
if __name__ == "__main__":
    words = get_random_medical_words()
    print(f"🧩 Gloss 단어들: {words}")
    sentence = generate_sentence_with_gpt(words)
    print(f"📝 생성된 문장: {sentence}")
