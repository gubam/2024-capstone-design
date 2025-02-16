"""
keypoint 추출 모듈
포인트 -> 칼만필터(추후 구현) -> 벡터화 -> 리턴
"""
import cv2
import numpy as np
import copy

# 0 : 오른손, 1 : 왼손, 2 : 상체
#frame은 원본 이미지, results는 미디어파이프 통과 데이터?

class keypoint():
    
    def __init__(self, mp_drawing, mp_holistic, holistic, kf_sw = True):
        self.mp_drawing = mp_drawing
        self.mp_holistic = mp_holistic
        self.holistic = holistic
        self.kf_sw = kf_sw
        
        self.pointDic = {
            "right" : [[0., 0., 0.] for _ in range(21)],
            "left" : [[0., 0., 0.] for _ in range(21)],
            "body" : [[0., 0., 0.] for _ in range(12)]}
        self.pre_pointDic ={
            "right" : [[0., 0., 0.] for _ in range(21)],
            "left" : [[0., 0., 0.] for _ in range(21)],
            "body" : [[0., 0., 0.] for _ in range(12)]}
        
        #sw = True면 칼만필터 적용
        
        self.kf_right = [KalmanFilterXY() for _ in range(21)]
        self.kf_left = [KalmanFilterXY() for _ in range(21)]
        self.kf_body = [KalmanFilterXY() for _ in range(12)]
    
    #주요 메서드
    def extract_keypoint(self, frame):
        
        #초기화
        self.pointDic = {}
        self.frame = frame
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)

        self.cv2_drawing_point(results)

        #추출한 좌표 리스트 "right" value값 넣기
        result = self.pointDic.items()
        
        data = list(result)

        return frame, data
    
            
    # 그리기 파트(cv2 이용), point값 추출            
    def cv2_drawing_point(self, results): 
            # 각 관절 좌표 그리기 (오른손, 왼손, 상체)
        
        # 오른손 랜드마크 그리기
        temp = []
        if results.right_hand_landmarks:
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                
                if(self.kf_sw):
                    x, y = self.kf_right[idx].update(landmark.x, landmark.y)
                else:
                    x, y = landmark.x, landmark.y
                    
                temp.append([landmark.x, landmark.y, landmark.z])
                x = int(x * self.frame.shape[1])  # 이미지 너비로 변환
                y = int(y * self.frame.shape[0])  # 이미지 높이로 변환  
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)  # 초록색 원
                
            self.pointDic["right"] = temp
            self.pre_pointDic["right"] = self.pointDic["right"]
                
        else:
            for i in range(20):
                if(self.kf_sw):
                    x, y = self.kf_right[i].update(
                        self.pre_pointDic["right"][i][0], 
                        self.pre_pointDic["right"][i][1]
                        )
                else:
                    x, y =self.pre_pointDic["right"][i][0], self.pre_pointDic["right"][i][1]
                    
                temp.append([x, y, float((self.pre_pointDic.get("right"))[i][2])])
                x = int(x * self.frame.shape[1])  # 이미지 너비로 변환
                y = int(y * self.frame.shape[0])  # 이미지 높이로 변환
                cv2.circle(self.frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # 초록색 원

                
        # 왼손 랜드마크 그리기
        temp = []
        if results.left_hand_landmarks:
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                
                if(self.kf_sw):
                    x, y = self.kf_left[idx].update(landmark.x, landmark.y)
                else:
                    x, y = landmark.x, landmark.y
                    
                temp.append([x, y, landmark.z])
                x = int(x * self.frame.shape[1])  # 이미지 너비로 변환
                y = int(y * self.frame.shape[0])  # 이미지 높이로 변환

                cv2.circle(self.frame, (x, y), 5, (255, 0, 0), -1)  # 파란색 원
                self.pointDic["left"] = temp
                self.pre_pointDic["left"] = self.pointDic["left"]
        else:
            for i in range(20):
                if(self.kf_sw):
                    x, y = self.kf_left[i].update(
                        float((self.pre_pointDic["left"])[i][0]), 
                        float((self.pre_pointDic["left"])[i][1])
                        )
                else:
                    x, y =self.pre_pointDic["left"][i][0], self.pre_pointDic["left"][i][1]
                    
                temp.append([x, y, float((self.pre_pointDic.get("left"))[i][2])])
                x = int(x * self.frame.shape[1])  # 이미지 너비로 변환
                y = int(y * self.frame.shape[0])  # 이미지 높이로 변환
                cv2.circle(self.frame, (int(x), int(y)), 5, (255, 0, 0), -1)  # 초록색 원
        

        
        # 상체 랜드마크 그리기 (11~22번)
        temp = []
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark[11:23]):
                
                if(self.kf_sw):
                    x, y = self.kf_body[idx].update(landmark.x, landmark.y)
                else:
                    x, y = landmark.x, landmark.y
                    
                temp.append([x, y, landmark.z])
                x = int(x * self.frame.shape[1])  # 이미지 너비로 변환
                y = int(y * self.frame.shape[0])  # 이미지 높이로 변환
                cv2.circle(self.frame, (x, y), 5, (0, 0, 255), -1)  # 빨간색 원
                self.pointDic["body"] = temp
                self.pre_pointDic["body"] = self.pointDic["body"]
        else:
            for i in range(12):
                if(self.kf_sw):
                    x, y = self.kf_body[i].update(
                        float((self.pre_pointDic["body"])[i][0]), 
                        float((self.pre_pointDic["body"])[i][1])
                        )
                else:
                    x, y =self.pre_pointDic["body"][i][0], self.pre_pointDic["body"][i][1]
                    
                temp.append([x, y, float((self.pre_pointDic.get("body"))[i][2])])
                x = int(x * self.frame.shape[1])  # 이미지 너비로 변환
                y = int(y * self.frame.shape[0])  # 이미지 높이로 변환
                cv2.circle(self.frame, (x, y), 5, (0, 0, 255), -1)  # 초록색 원
        
        
        
        #def kalmanfilter():
            
        # 추출한 포인트들 벡터화 및 크기 1로 변환, depth 데이터 변환
        #def vectoriaztion(keypoint):

class KalmanFilterXY:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # 4개 상태 변수(x, y, vx, vy), 2개 측정 변수(x, y)
        dt = 1  # 시간 간격 (프레임 단위)

        # 상태 전이 행렬 (State Transition Matrix)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 측정 행렬 (Measurement Matrix)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 프로세스 노이즈 공분산 행렬 (Process Noise Covariance)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # 측정 노이즈 공분산 행렬 (Measurement Noise Covariance)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        # 초기 상태값 (State Post)
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

        # 🔥 이전 좌표를 저장 (손이 사라졌을 때 사용)
        self.last_x, self.last_y = None, None

    def update(self, x, y):
        
        predicted = self.kf.predict()  # 예측 단계

        if x == -1 and y == -1:  # 손실된 키포인트
            if self.last_x is not None:
                x, y = self.last_x, self.last_y  # 이전 값 사용
            else:
                return predicted[0, 0], predicted[1, 0]  # 예측값 사용

        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kf.correct(measurement)  # 보정 단계

        # 🔥 최신 좌표 업데이트
        self.last_x, self.last_y = corrected[0, 0], corrected[1, 0]

        return corrected[0, 0], corrected[1, 0]  # 보정된 (x, y) 반환
