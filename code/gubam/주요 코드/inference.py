from model import ModelLoader
import mp_keypoint
import training
import cv2
import math

VIDEO_SRC = "C:/Users/82109/Downloads/KakaoTalk_20250312_141834806.mp4"

#인스턴스 선언 부분
keypoint = mp_keypoint.keypoint(kf_sw = True, draw_graph_sw = False, z_kill = True)
cap = cv2.VideoCapture(VIDEO_SRC)
sampling = training.ScoreSampling(50)
model = ModelLoader('lstm')

#최대 길이 제한용
count = 0

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

length = len(score)  # score_list의 길이 기준으로 자름
chunk_size = 100
num_chunks = math.ceil(length / chunk_size)  # 나눠 떨어지지 않으면 마지막 덩어리도 포함
print(length)

idx = 0
angle_temp =[]
for i in range(num_chunks):
    score_chunk = score[idx:idx+chunk_size]
    frame_chunk = frame[idx:idx+chunk_size]
    angle_chunk = angle[idx:idx+chunk_size]

    frame_list, angle_list = sampling.sampling(score_chunk, frame_chunk, angle_chunk)
    angle_temp.append(angle_list)

    for f in frame_list:
        cv2.imshow("Sampling Video", cv2.resize(f, dsize=(960, 540)))
        if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
            break

    idx += chunk_size

cap.release()
cv2.destroyAllWindows()

for i in range(len(angle_temp)):
    output = model.inference_output(angle_temp[i])
    print(output)