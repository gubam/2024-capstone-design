import cv2
import mp_keypoint
import verification

'''
영상 경로
json 저장 경로
json 저장 폴더
'''
VIDEO_SRC = "C:/Users/gubam/Desktop/수어 데이터셋/원본영상/나_WORD1157/NIA_SL_WORD1157_REAL06_F.mp4"
SAVE_PATH = "C:/Users/gubam/Desktop/json"
FOLDER_NAME = "왼쪽"
#미디어파이프 인스턴스 선언 부분


keypoint = mp_keypoint.keypoint(kf_sw = True)
save =  mp_keypoint.SaveJson(SAVE_PATH, FOLDER_NAME)
cap = cv2.VideoCapture(VIDEO_SRC)
ver = verification.verification()


'''
keykeypoint.flatvec 이건 추출 벡터값이고
keypoint.angle 이건 추출되는 각도값임
순서는 오른손 -> 왼손 -> 몸 - > Z방향 값
'''
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image = keypoint.extract_keypoint(frame)
    output = keypoint.flatvec
    print(keypoint.angle)
    ver.angle_ver(keypoint.angle)
    #print(output)
    #save.save_data(output)
    cv2.imshow('video',cv2.resize(image,dsize=(960,540)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
