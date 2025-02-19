def mp_drawing_point(self, results):
        
        temp = []
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                self.frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                temp.append([landmark.x, landmark.y, landmark.z])
        else:
            for i in range(20):
                temp.append([-1, -1, -1])
        self.pointDic["right"] = temp
        
        temp = []
        # 왼손 랜드마크 처리
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                self.frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                temp.append([landmark.x, landmark.y, landmark.z])
        else:
            for i in range(20):
                temp.append([-1, -1, -1])
        self.pointDic["left"] = temp
        
        temp = []
        # 바디 랜드마크 처리 (11-22번)
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark[11:23]):
                temp.append([landmark.x, landmark.y, landmark.z])
            self.mp_drawing.draw_landmarks(
                self.frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        else:
            for i in range(12):
                temp.append([-1, -1, -1])
            self.pointDic["body"] = temp