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
import matplotlib.pyplot as plt

class keypoint:
    '''
    파라미터각 인스턴스 넣어주기
    kf_sw는 칼만필터 on/off
    출력은 frame과 단일 프레임 벡터 좌표
    output -> right: 42, left: 42 body: 24
    '''

    def __init__(self, 
                 holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9), 
                 mp_holistic = mp.solutions.holistic,  
                 kf_sw = True, 
                 draw_graph_sw = True, 
                 z_kill = True
                 ):
        
        #설정값 스위치
        self.kf_sw = kf_sw
        self.draw_graph_sw = draw_graph_sw
        self.z_kill = z_kill
        
        self.mp_holistic = mp_holistic
        self.holistic = holistic
        
        self.mp_holistic = mp_holistic
        self.holistic = holistic
               
        #그래프 관련
        if draw_graph_sw ==True:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 5))
            self.fig.canvas.manager.set_window_title("hand & body angle data")
            if z_kill == False:
                self.fig2, self.ax2 = plt.subplots(figsize=(12, 5))

        
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
        self.hand_index = None


        ###주로 사용###
        self.score_list = []
        self.angle_list = []
        self.frame_list = []
        self.score = 0        
        self.pre_flatvec = []
        self.flatvec = []
        self.angle = []
        self.pre_angle = []

    
    #주요 메서드
    def extract_keypoint(self, frame):
        self.frame = frame
        #좌표추출
        self._cv2_drawing_point(frame)
        #벡터로 변환 angle 업데이트
        self._vectorization(self.pointDic)
        
        if self.draw_graph_sw == True:
            self.__draw_graph()
        
        return


    def __draw_graph(self):
        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[2].clear()
        
        
        x1 = np.arange(20)
        x2 = np.arange(20, 40)
        x3 = np.arange(40, 50)  # 여기도 20~40으로 설정 (기존 코드 유지)
        
        right_hand = self.angle[:20]
        left_hand = self.angle[20:40]
        body = self.angle[40:50]
        
        #z값 관련
        if not self.z_kill:
            x4 = np.arange(51, 102)
            self.ax2.clear()
            z_data = self.angle[50:]
            self.ax2.bar(x4, z_data)
            self.ax2.set_ylim(0, 1)
            self.ax2.set_title("z data")
        
        # 그래프 업데이트
        self.ax[0].bar(x1, right_hand)
        self.ax[1].bar(x2, left_hand)
        self.ax[2].bar(x3, body) 
        
        self.ax[0].set_ylim(0, 3.2)
        self.ax[1].set_ylim(0, 3.2)
        self.ax[2].set_ylim(0, 3.2)
        
        # 제목 및 범례 추가
        self.ax[0].set_title("right hand")
        self.ax[1].set_title("left hand")
        self.ax[2].set_title("body")
        
        plt.tight_layout()
        plt.show()
        plt.pause(0.01)
        return


    def __initialization(self):
        self.intial = True
       

    # 그리기 파트(cv2 이용), point값 추출            
    def _cv2_drawing_point(self, results): 
            # 각 관절 좌표 그리기 (오른손, 왼손, 상체)
        image = cv2.cvtColor(results, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        
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
           
    # 추출한 포인트들 벡터화 및 크기 1로 변환, depth 데이터 변환 최종적으로는 클래스 변수값 저장 + 스코어까지 내기
    def _vectorization(self, keypoint):
        self.pre_angle = self.angle  
        self.pre_flatvec = self.flatvec     
        self.Z_data = []
        output =[]
        right = []
        left = []
        body = []
        right.append(self._hand_vector(keypoint["right"]))
        left.append(self._hand_vector(keypoint["left"]))
        body.append(self._body_vector(keypoint["body"]))
        right = right[0]
        left = left[0]
        body = body[0]
        right.extend(self._hand_vector_2(keypoint["right"],shoulder_point=keypoint["body"][1]))
        left.extend(self._hand_vector_2(keypoint["left"], shoulder_point=keypoint["body"][0]))
        body.extend(self._body_vector_2(keypoint["body"]))
                
        output = (self.__angle(right, left, body))
        
        if not self.z_kill:
            output.extend(self.Z_data[0])
            output.extend(self.Z_data[1])
            output.extend(self.Z_data[2])

        self.angle = output
        self.__score()
        if self.score:
            self.score_list.append(self.score)
            self.angle_list.append(self.angle)
            self.frame_list.append(self.frame)

        return
        
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

    # 어깨 기준 벡터 생성
    def _hand_vector_2(self, hand_point,shoulder_point):
        output = []
        Z_output = []
        finger_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        for i in range(len(finger_list)):
            point2_idx = finger_list[i]
            output.append(self._unit_vector( self._vector_XY(shoulder_point, hand_point[point2_idx])))

            if hand_point[i][2] < 0:
                Z_output.append(0)
            else:
                Z_output.append(1)
            
        self.Z_data.append(Z_output)
        
        return output
    
    def _body_vector_2(self, body_point):
        output = []
        Z_output = []
        body_list= [[0,2],[1,3],[0,4],[1,5],[0,6],[1,7],[0,8],[1,9],[0,10],[1,11]]
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
    


    def __score(self):
        sum = 0
        if not self.pre_angle:
            return 
        for i in range(len(self.angle)):
            sum += abs(self.angle[i] - self.pre_angle[i])
        self.score = float(f"{(sum):.7f}")
        return
    
    #현재 프레임의 각도 추출
    def __angle(self, right, left, body):
        std = body[0]
        right_ag = []
        left_ag = []
        body_ag =[]
        output = []
        self.flatvec.append(right)
        self.flatvec.extend(left)
        self.flatvec.extend(body)
        for i in range(len(right)):
            dot = std[0] * right[i][0] + std[1] * right[i][1]
            right_ag.append( float(f"{(np.arccos(np.clip(dot, -1.0, 1.0))):.7f}" ))
        for i in range(len(left)):
            dot = std[0] * left[i][0] + std[1] * left[i][1]
            left_ag.append(float(f"{(np.arccos(np.clip(dot, -1.0, 1.0))):.7f}" ))
        for i in range(1,len(body)):
            dot = std[0] * body[i][0] + std[1] * body[i][1]
            body_ag.append(float(f"{(np.arccos(np.clip(dot, -1.0, 1.0))):.7f}" )) 
        output.append(right_ag)
        output = output[0]
        output.extend(left_ag)
        output.extend(body_ag) 
        return output

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