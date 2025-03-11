import cv2
import mp_keypoint
import verification
import training
'''
VIDEO_SRC = 영상 경로
SAVE_PATH = json 저장 경로
FOLDER_NAME = json 저장 폴더(폴더 안에0,1,2,3...이름의 json이 생성됨)
'''
VIDEO_SRC = "C:/Users/82109/Desktop/수어영상/나_WORD1157.mp4"
SAVE_PATH = "C:/Users/gubam/Desktop/json"
FOLDER_NAME = "왼쪽"

#인스턴스 선언 부분
keypoint = mp_keypoint.keypoint(kf_sw = True)
save =  mp_keypoint.SaveJson(SAVE_PATH, FOLDER_NAME)
cap = cv2.VideoCapture(VIDEO_SRC)
ver = verification.verification()
sampling = training.ScoreSampling(100)

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
    #스코어 비교 인스턴스
    sampling.sampling(keypoint)

    # print((sampling.get_samples()))
    #print(output)
 
    '''
    아래 주석 제거하면 angle값 그래프 출력
    '''
    ver.angle_ver(keypoint.angle)
    
    '''
    아래 주석 제거하면 json 경로에 output()저장
    '''
    #save.save_data(output)

    cv2.imshow( 'video', cv2.resize(image,dsize=(960,540)) )
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

