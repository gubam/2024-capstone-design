'''
데이터셋의 json 파일에서
원하는 단어 매칭하는 코드
word dataset/morpheme/sen와 word의 파일 경로 알맞게 수정하기
원하는 단어 target_word 설정하기
'''

import os
import json

# 검색할 단어 설정
target_word = "이마"

# JSON 파일이 저장된 디렉토리 경로
directory = "dataset/morpheme/sen/02"


found_files = []

# 디렉토리 내의 모든 파일에 대해 반복
for filename in os.listdir(directory):
    if filename.endswith(".json"):  # 확장자가 .json인 파일만 처리
        file_path = os.path.join(directory, filename)

        # 파일을 열고 JSON 데이터 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)  # JSON 파싱
                
                # 'data' 리스트에서 'attributes'의 'name' 값 확인
                if "data" in data:
                    for item in data["data"]:
                        if "attributes" in item:
                            for attribute in item["attributes"]:
                                if "name" in attribute and attribute["name"] == target_word:
                                    found_files.append(filename)  # 파일명 저장
                                    break  # 단어를 찾으면 그 파일은 더 이상 확인할 필요 없음
            except json.JSONDecodeError:
                print(f"Error reading {filename}")

# 검색 결과 출력
print(f"Found {len(found_files)} files containing the word '{target_word}' in attributes name:")
for file in found_files:
    print(file[7:14])


directory = "C:/Users/gubam/Desktop/word/1"

found_files = []

# 디렉토리 내의 모든 파일에 대해 반복
for filename in os.listdir(directory):
    if filename.endswith(".json"):  # 확장자가 .json인 파일만 처리
        file_path = os.path.join(directory, filename)

        # 파일을 열고 JSON 데이터 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)  # JSON 파싱
                
                # 'data' 리스트에서 'attributes'의 'name' 값 확인
                if "data" in data:
                    for item in data["data"]:
                        if "attributes" in item:
                            for attribute in item["attributes"]:
                                if "name" in attribute and attribute["name"] == target_word:
                                    found_files.append(filename)  # 파일명 저장
                                    break  # 단어를 찾으면 그 파일은 더 이상 확인할 필요 없음
            except json.JSONDecodeError:
                print(f"Error reading {filename}")

# 검색 결과 출력
print(f"Found {len(found_files)} files containing the word '{target_word}' in attributes name:")
for file in found_files:
    print(file)


