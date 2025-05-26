from openai import OpenAI

# ✅ OpenAI API 키 설정
client = OpenAI(api_key="")

#gloss map 그리기
gloss_map = ["감사합니다","골절", "다리","있다", "뜨겁다", "아프다", "안녕하세요", "어디", "오다", "주다", "진단서", "체온"]

# GPT에게 문장 생성 요청
# person은 의사, 환자 두가지 상황을 제시함
def generate_sentence_with_gpt(words, person):
    words = remove_duplicates(words)
    prompt = (
        f"다음은 수어의 gloss 단어들입니다. "
        f"이 단어들을 사용하여 병원 현장에서 {person}가 사용할 수 있는 구어체 한국어 문장으로 바꿔주세요.\n\n"
        f"🔹 규칙:\n"
        f"- 주어진 단어만 사용 (단어 순서는 재배치 가능)\n"
        f"- 조사나 어미는 자연스럽게 붙여도 됨\n"
        f"- 새로운 단어나 동사는 절대 추가하지 말 것\n"
        f"- 자연스러운 구어체 문장으로 작성\n\n"
        f"📝 단어 목록: {', '.join(words)}"
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
        f"해당 gloss 맵에 없는 단어는 절대 추출하지말고 알맞은 글로스들을 추출해줘"
        f"출력의 형식은 무조건 순서대로 콤마로 구분주고 공백은 만들면 안되"
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