import heapq

class ScoreSampling:
    def __init__(self, input_len, skip_sample=10):
        self.input_len = input_len  # 최대 저장 개수
        self.skip_sample = skip_sample  # 처음 무시할 개수
        self.counter = 0  # 전체 입력 개수 카운트
        self.min_score = float('inf')  # 최소 스코어 추적
        self.out_list = []  # 샘플링된 데이터 저장
        self.score_heap = []  # (스코어, 리스트 내 인덱스) 최소 힙으로 유지
    
    def sampling(self, keypoint):
        score = keypoint.score
        data = keypoint.angle

        # 스킵 샘플링 구간에서는 저장하지 않음
        if self.counter < self.skip_sample:
            self.counter += 1
            return

        # 리스트가 가득 찬 경우, 최소 스코어와 비교하여 교체
        if len(self.out_list) == self.input_len:
            if score > self.min_score:
                # 최소값을 가진 요소 제거
                heapq.heappop(self.score_heap)  
                self.out_list.pop(0)  # 리스트 맨 앞 요소 제거 (FIFO 방식)
                
                # 새로운 데이터를 리스트 끝에 추가
                self.out_list.append(data)
                heapq.heappush(self.score_heap, (score, self.counter))  # 새로운 데이터 추가
                self.min_score = self.score_heap[0][0]  # 최소 스코어 갱신
        else:
            # 아직 리스트가 다 차지 않았다면 그냥 삽입
            self.out_list.append(data)
            heapq.heappush(self.score_heap, (score, self.counter))
            self.min_score = self.score_heap[0][0]  # 최소 스코어 갱신

        self.counter += 1

    def get_samples(self):
        return self.out_list
