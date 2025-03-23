import cv2
import mp_keypoint
import training
'''
VIDEO_SRC = 영상 경로
SAVE_PATH = json 저장 경로
FOLDER_NAME = json 저장 폴더(폴더 안에0,1,2,3...이름의 json이 생성됨)
'''
VIDEO_SRC = "C:/Users/gubam/Desktop/수어 데이터셋/원본영상/간호사_WORD0187/NIA_SL_WORD0187_REAL02_F.mp4"
SAVE_PATH = "C:/Users/gubam/Desktop/json"
FOLDER_NAME = "오른쪽"

#인스턴스 선언 부분
keypoint = mp_keypoint.keypoint(kf_sw = False, draw_graph_sw = True, z_kill = True)
cap = cv2.VideoCapture(VIDEO_SRC)
#sampling = training.ScoreSampling(100)
counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    keypoint.extract_keypoint(frame)
    image = keypoint.frame
    output = keypoint.flatvec
    #sampling.sampling(keypoint)
    
    cv2.imshow( 'video', cv2.resize(image,dsize=(960,540)) )
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    counter += 1
'''
아래 주석 제거하면 json 경로에 output()저장
'''    
#save.save_data(sampling.get_samples())

cap.release()
cv2.destroyAllWindows()