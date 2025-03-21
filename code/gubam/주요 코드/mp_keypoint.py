"""
keypoint 추출 모듈
포인트 -> 칼만필터(추후 구현) -> 벡터화 -> 리턴
z 축은 hand만 추출 0, 1가짐 앞,뒤 기억안나(txt 파일 확인)
내부 extract_keypointrk 주요함수
해당함수의 출력은 frame과 벡터화된 좌표
추후 할일 
1. 초기값 설정
2. 모션 스코어를 이용한 샘플링
"""
import cv2
import numpy as np
import math
import mediapipe as mp
import json
import os

# 0 : 오른손, 1 : 왼손, 2 : 상체
class keypoint:
    '''
    파라미터각 인스턴스 넣어주기
    kf_sw는 칼만필터 on/off
    출력은 frame과 단일 프레임 벡터 좌표
    output -> right: 42, left: 42 body: 24
    '''

    def __init__(self, kf_sw = True):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.kf_sw = kf_sw
        # 추후 초기값 결정 pre 값은 추출안될때(순수 포인트값)
        self.pointDic = {
            "right" : [[0.1, 0.1, 0.1] for _ in range(21)],
            "left" : [[0.1, 0.1, 0.1] for _ in range(21)],
            "body" : [[0.1, 0.1, 0.1] for _ in range(12)]}
        self.pre_pointDic ={
            "right" : [[0.1, 0.1, 0.1] for _ in range(21)],
            "left" : [[0.1, 0.1, 0.1] for _ in range(21)],
            "body" : [[0.1, 0.1, 0.1] for _ in range(12)]}
        
        #sw = True면 칼만필터 적용
        
        self.kf_right = [KalmanFilterXY() for _ in range(21)]
        self.kf_left = [KalmanFilterXY() for _ in range(21)]
        self.kf_body = [KalmanFilterXY() for _ in range(12)]
        
        self.initial = True
        
        self.Z_data = []

        ###주로 사용###
        self.score = 0        
        self.pre_flatvec = []
        self.flatvec = []
        self.angle = []

    
    #주요 메서드
    def extract_keypoint(self, frame):

        if self.initial:
            #초기화
            self.frame = frame
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)

            self._cv2_drawing_point(results)

            #아래는 다 정리하기
            output = self._vectorization(self.pointDic)
            pre_output = self._vectorization(self.pre_pointDic)
            
            pre_output = self._flatten(pre_output)
            output = self._flatten(output)
            
            #self.score = self.__score(output, pre_output)
            self.pre_flatvec = pre_output
            self.flatvec = output
            
            self.__angle()
            return frame
        
        else:
            self.__initialization()

        

    # 그리기 파트(cv2 이용), point값 추출            
    def _cv2_drawing_point(self, results): 
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
                
            self.pre_pointDic["right"] = self.pointDic["right"]    
            self.pointDic["right"] = temp
            
                
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
                self.pre_pointDic["left"] = self.pointDic["left"]
                self.pointDic["left"] = temp
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
                self.pre_pointDic["body"] = self.pointDic["body"]
                self.pointDic["body"] = temp
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
           
    # 추출한 포인트들 벡터화 및 크기 1로 변환, depth 데이터 변환
    def _vectorization(self, keypoint):       
        self.Z_data = []
        output =[]
        right = []
        left = []
        body = []
        right.append(self._hand_vector(keypoint["right"]))
        left.append(self._hand_vector(keypoint["left"]))
        body.append(self._body_vector(keypoint["body"]))
        print(right)
        print(left)
        print(body)
        output.append(self.Z_data)
        return output
        
    # x,y,z 포인트 21개 리스트로 들어옴 21 * 3
    def _hand_vector(self, hand_point):
        output = []
        Z_output = []
        finger_list = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
        for i in range(len(finger_list)):
            point1_idx = finger_list[i][0]
            point2_idx = finger_list[i][1]
            output.append(self._unit_vector( self._vector_XY(hand_point[point1_idx], hand_point[point2_idx])))

            if hand_point[i][2] < 0:
                Z_output.append(0)
            else:
                Z_output.append(1)
            
        self.Z_data.append(Z_output)
        
        return output
    
    def _body_vector(self, body_point):
        output = []
        Z_output = []
        body_list= [[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[4,10],[5,11],[4,8],[5,9]]
        for i in range(len(body_list)):
            point1_idx = body_list[i][0]
            point2_idx = body_list[i][1]
            output.append(self._unit_vector( self._vector_XY(body_point[point1_idx], body_point[point2_idx])))
            
            if body_point[i][2] < 0:
                Z_output.append(0)
            else:
                Z_output.append(1)

        self.Z_data.append(Z_output)
        
        return output          
        
    #두개의 포인트 입력하면 두 점의 XY 벡터 추출
    def _vector_XY(self, point1, point2):
        outputXY = [0, 0]
        outputXY[0] = point2[0] - point1[0]
        outputXY[1] = point2[1] - point1[1]   
        return outputXY

    #vector input [0] = x, [1] = y
    def _unit_vector(self, vector):
        x, y = vector[0], vector[1]
        mag = math.sqrt( x**2 + y**2 )
        if(mag == 0 ):
            mag = 0.000001
        unit_x = float(f"{(x / mag):.7f}")
        unit_y = float(f"{(y / mag):.7f}")
        output = [unit_x, unit_y]
        return output
    
    def _flatten(self, list):
        output = []
        for list in enumerate(list):
            temp = list[1]
            for temp in enumerate(temp):
                temp = temp[1]
                output.append(temp[0])
                output.append(temp[1])
        return output
    
    def __initialization(self):
        self.intial = True
       

    def __score(self, point ,pre_point):
        sum = 0
        for i in range(108):
            sum += (point[i] - pre_point[i])
        sum = abs(sum)
        float(f"{(sum):.7f}")
        return sum
    
    #현재 프레임의 각도 추출
    def __angle(self):
        vec = self.flatvec
        output = []
        for i in range(0, 106, 2):
            x1, x2 = vec[i], vec[i+2]
            y1, y2 = vec[i+1], vec[i+3]
            dot = x1 * x2 + y1 * y2
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            output.append(angle)

        self.angle = output

    #각도 프레임별 차이 추출
    def __anglediff(self):
        vec = self.flatvec
        prevec = self.pre_flatvec

        output = []
        for i in range(0, 108, 2):
            x1, x2 = prevec[i], vec[i]
            y1, y2 = prevec[i+1], vec[i+1]
            dot = x1 * x2 + y1 * y2
            
            angle = np.arccos(np.clip(dot, -1.0, 1.0))
            
            output.append(angle)
        self.angle = output
            
        
class SaveJson:
    '''
    경로, 폴더 이름 입력으로 넣기
    '''
    
    def __init__(self, SAVE_PATH, FOLDER_NAME, Y_label):
        self.count = 1
        self.SAVE_PATH = SAVE_PATH
        self.FOLDER_NAME = FOLDER_NAME
        self.Y_label = Y_label
        self.directory = f"{self.SAVE_PATH}/{self.FOLDER_NAME}"

    def save_data(self, data, name):
        filename = f"{self.directory}/{name}.json"
        
        if not os.path.exists(self.directory):  # 디렉토리가 없으면 생성
            os.mkdir(self.directory)

        jsondata = {"X": data, "Y" : self.Y_label}
        with open(filename, "w", encoding="utf-8") as outfile:
            json.dump(jsondata, outfile, indent=4, ensure_ascii=False)
        self.count += 1
    
        
#GPT가 작성 잘모름
class KalmanFilterXY:
    
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # 4개 상태 변수(x, y, vx, vy), 2개 측정 변수(x, y)
        dt = 30  # 시간 간격 (프레임 단위)

        # 상태 전이 행렬 
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 측정 행렬
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 프로세스 노이즈 공분산 행렬
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # 측정 노이즈 공분산 행렬
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        # 초기 상태값
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

        # 이전 좌표를 저장 (손이 사라졌을 때 사용)
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


        return corrected[0, 0], corrected[1, 0]  # 보정된 (x, y) 반환