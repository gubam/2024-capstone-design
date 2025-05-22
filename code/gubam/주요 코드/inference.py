from model import ModelLoader
import mp_keypoint
# import training
import cv2
import math
# import gpt_api
while True:
    VIDEO_SRC = "C:/Users/82109/Downloads/회복_WORD0057.mp4"
    #인스턴스 선언 부분
    keypoint = mp_keypoint.keypoint(kf_sw = True, draw_graph_sw = False, z_kill = True)
    cap = cv2.VideoCapture(VIDEO_SRC)
    # sampling = training.ScoreSampling(50 ,skip_sample=0)
    # model = ModelLoader('lstm')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoint.extract_keypoint(frame)
        image = keypoint.frame
        output = keypoint.angle
        
        cv2.imshow( 'video', cv2.resize(image,dsize=(960,540)) )

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
score = keypoint.score_list
frame = keypoint.frame_list
angle = keypoint.angle_list

# length = len(score)  # score_list의 길이 기준으로 자름
# sample_size = 90
# num_sample = math.ceil((length-100) / 10 )

# print(length)
# print(num_sample)

# idx = 0
# angle_temp =[]
# for i in range(2):
#     score_chunk = score[idx:idx+sample_size]
#     frame_chunk = frame[idx:idx+sample_size]
#     angle_chunk = angle[idx:idx+sample_size]

#     frame_list, angle_list = sampling.sampling(score_chunk, frame_chunk, angle_chunk)
#     angle_temp.append(angle_list)

    # # 샘플링영상 켜기
    # for f in frame_list:
    #     cv2.imshow("Sampling Video", cv2.resize(f, dsize=(960, 540)))
    #     if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
    #         break

#     idx += 10

# cap.release()
# cv2.destroyAllWindows()
# temp =[]
# for i in range(len(angle_temp)):
#     output = model.inference_output(angle_temp[i])
#     temp.append(output)
#     print(output)

# gpt_out = gpt_api.generate_sentence_with_gpt(temp)
# print(gpt_out)

# gpt_api.tts_output(gpt_out)