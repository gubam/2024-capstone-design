'''
어떤식으로 코드 작성을 해야하지
그냥 스코어기준 샘플링
'''
class ScoreSampling:
    def __init__(self, sampling_len, skip_sample = 10):
        self.sampling_len = sampling_len  # 최대 저장 개수
        self.skip_sample = skip_sample  # 처음 무시할 개수
        self.loop_counter = 0

    def sampling(self, score_list, frame_list, angle_list):
        # 스킵 샘플링 구간에서는 저장하지 않음
        sorted_indices = [i for i, _ in sorted(enumerate(score_list), key=lambda x: x[1])]

        if self.sampling_len < len(score_list):
            while self.sampling_len < len(score_list):
                idx = sorted_indices[0]
                del score_list[idx]
                del frame_list[idx]
                del angle_list[idx]
                sorted_indices = [i for i, _ in sorted(enumerate(score_list), key=lambda x: x[1])]

            return frame_list, angle_list
        else:
            return frame_list, angle_list

# #100개 중 50개 샘플링            
# class VideoWindow:
#     def __init__(self, input_frame, sampling_frame = 100, stride = 10):
#         self.input_frame = input_frame
#         self.sampling_frame = sampling_frame
#         self.stride = stride
#         self.video_len = len(input_frame)
#         self.pre_idx = 0
        
#     def VideoSampling(self):
#         pass

            


