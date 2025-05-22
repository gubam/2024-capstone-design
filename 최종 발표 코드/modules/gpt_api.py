from openai import OpenAI

# ✅ OpenAI API 키 설정
client = OpenAI(api_key="")

#gloss map 그리기
gloss_map = ["안녕하세요","감사합니다","나","아프다","머리","손","이다"]

# GPT에게 문장 생성 요청
# person은 의사, 환자 두가지 상황을 제시함
def generate_sentence_with_gpt(words, person):
    words = remove_duplicates(words)
    prompt = (
        f"다음 단어들은 수어의 gloss입니다. "
        f"이 단어들만 사용해서 자연스럽고 문법에 맞는 한국어 문장으로 바꿔줘. "
        f"필요하다면 조사나 어미는 붙여도 되지만, 새로운 단어나 동사는 절대 추가하지 마. "
        f"문장은 병원현장에서 {person}가 쓸만한 구어체로 문장으로 작성을 해줘. "
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


def remove_duplicates(words):
    return list(set(words))

def generate_gloss_with_gpt(sentence):
    prompt = (
        f"다음 문장에서 gloss로 변환해야해"
        f"우리가 가진 gloss의 맵은 {gloss_map}이고"
        f"해당 gloss 맵에 없는 단어는 추출하지말고 알맞은 글로스들을 추출해줘"
        f"출력의 형식은 무조건 순서대로 콤마로 구분해줘"
        f"문장은 {sentence}야"
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