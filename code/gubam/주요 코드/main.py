import cv2
import mp_keypoint

VIDEO_SRC = "C:/Users/gubam/Desktop/수어 데이터셋/원본영상/왼쪽_WORD1342.mp4"
SAVE_PATH = "C:/Users/gubam/Desktop/json"
FOLDER_NAME = "왼쪽"
#미디어파이프 인스턴스 선언 부분

keypoint = mp_keypoint.keypoint(kf_sw = True)
save =  mp_keypoint.SaveJson(SAVE_PATH, FOLDER_NAME)
cap = cv2.VideoCapture(VIDEO_SRC)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image, output = keypoint.extract_keypoint(frame)
    
    print(keypoint.score)
    # print()
    #save.save_data(output)
    cv2.imshow('video',cv2.resize(image,dsize=(960,540)))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
