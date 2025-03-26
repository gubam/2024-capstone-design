import cv2
import mp_keypoint
import training
'''
VIDEO_SRC = 영상 경로
SAVE_PATH = json 저장 경로
FOLDER_NAME = json 저장 폴더(폴더 안에0,1,2,3...이름의 json이 생성됨)
'''
VIDEO_SRC = "C:/Users/82109/Desktop/데이터셋/수어영상 데이터셋/머리_WORD1249/NIA_SL_WORD1249_REAL02_F.mp4"
SAVE_PATH = "C:/Users/gubam/Desktop/json"
FOLDER_NAME = "오른쪽"

#인스턴스 선언 부분
keypoint = mp_keypoint.keypoint(kf_sw = True, draw_graph_sw = False, z_kill = True)
cap = cv2.VideoCapture(VIDEO_SRC)
sampling = training.ScoreSampling(20)
counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    keypoint.extract_keypoint(frame)
    image = keypoint.frame
    output = keypoint.angle
    #print(len(keypoint.angle_list))
    
    cv2.imshow( 'video', cv2.resize(image,dsize=(960,540)) )
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    counter += 1

'''
아래 주석 제거하면 json 경로에 output()저장
'''    
#save.save_data(sampling.get_samples())
# print(keypoint.angle_list)
# print(len(keypoint.angle_list))
frame_list, angle_list = sampling.sampling(keypoint.score_list,keypoint.frame_list,keypoint.angle_list)
print(len(frame_list))
for frame in frame_list:
    cv2.imshow("Video Playback",cv2.resize(frame,dsize=(960,540)) )  # 이미지 표시
    if cv2.waitKey(int(1/30 * 1000)) & 0xFF == ord('q'):  # 'q'를 누르면 종료
        break

print(len(angle_list))
cap.release()
cv2.destroyAllWindows()