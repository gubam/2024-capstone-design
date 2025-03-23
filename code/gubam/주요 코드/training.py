'''
어떤식으로 코드 작성을 해야하지
그냥 스코어기준 샘플링
'''
class ScoreSampling:
    def __init__(self, input_len, skip_sample=10):
        self.input_len = input_len  # 최대 저장 개수
        self.skip_sample = skip_sample  # 처음 무시할 개수
        self.loop_counter = 0

    def sampling(self, score_list, frame_list):
        # 스킵 샘플링 구간에서는 저장하지 않음
        min_score = float('inf')
        
        for i in range(len(score_list)):
            if i < self.input_len:
                
            
            


