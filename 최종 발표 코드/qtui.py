import sys
import time
import cv2
import mediapipe as mp
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer
from ui_modules.vision_utils import detect_face, detect_hands
from ui_modules.display_utils import display_frame


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("수어 번역 시스템")
        self.setMinimumSize(1000, 600)

        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
        self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0)

        self.cap = cv2.VideoCapture(0)
        self.use_webcam = True

        self.condition_start_time = None
        self.event_triggered = False

        self.init_ui()
        self.init_timer()

    def init_ui(self):
        self.label_video = QLabel("영상 영역")
        self.label_video.setStyleSheet("background-color: #cccccc;")
        self.label_video.setMinimumSize(600, 400)
        self.label_video.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_mode = QPushButton("모드변경")
        self.btn_record = QPushButton("녹음")
        self.btn_mode.clicked.connect(self.toggle_mode)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_mode)
        btn_layout.addWidget(self.btn_record)

        self.label_status = QLabel()
        self.label_status.setFixedSize(40, 40)
        self.label_status.setStyleSheet("background-color: red; border-radius: 20px;")

        self.message_box = QTextEdit()
        self.message_box.setPlaceholderText("message...")
        self.message_box.setMinimumHeight(200)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.label_video)

        right_layout = QVBoxLayout()
        right_layout.addLayout(btn_layout)
        right_layout.addWidget(self.label_status, alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.message_box)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)

        self.setLayout(main_layout)

    def toggle_mode(self):
        self.use_webcam = not self.use_webcam

        if self.use_webcam:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
        else:
            self.timer.stop()
            if self.cap.isOpened():
                self.cap.release()

            image_path = "C:/Users/82109/Pictures/Screenshots/test.png"
            image = cv2.imread(image_path)
            if image is not None:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                display_frame(self.label_video, rgb)
                self.label_status.setStyleSheet("background-color: gray; border-radius: 20px;")
            else:
                self.message_box.setText("이미지를 불러올 수 없습니다.")

    def init_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        if not self.use_webcam:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height = frame.shape[0]

        face_ok = detect_face(rgb, self.mp_face)
        left_ok, right_ok = detect_hands(rgb, self.mp_hands)

        # 중심 좌표 추출
        face_results = self.mp_face.process(rgb)
        hand_results = self.mp_hands.process(rgb)
        face_y = self.get_face_center_y(face_results, frame_height)
        hand_y = self.get_hand_center_y(hand_results, frame_height)

        # 조건 유지 확인 및 이벤트 트리거
        self.check_pose_hold_and_trigger_event(face_y, hand_y, frame_height)

        self.update_led_status(face_ok, left_ok, right_ok)
        display_frame(self.label_video, rgb)

    def update_led_status(self, face_ok, left_ok, right_ok):
        color = "green" if face_ok and left_ok and right_ok else "red"
        self.label_status.setStyleSheet(f"background-color: {color}; border-radius: 20px;")

    def get_face_center_y(self, results, frame_height):
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            center_y = bbox.ymin + bbox.height / 2
            return int(center_y * frame_height)
        return None

    def get_hand_center_y(self, results, frame_height):
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                ys = [lm.y for lm in hand.landmark]
                avg_y = sum(ys) / len(ys)
                return int(avg_y * frame_height)
        return None

    def check_pose_hold_and_trigger_event(self, face_y, hand_y, frame_height):
        face_top = face_y is not None and face_y < frame_height / 3
        hand_middle = hand_y is not None and (frame_height / 3) < hand_y < (frame_height * 2 / 3)

        if face_top and hand_middle:
            if self.condition_start_time is None:
                self.condition_start_time = time.time()
            elif time.time() - self.condition_start_time >= 3 and not self.event_triggered:
                self.event_triggered = True
                self.on_pose_held_event()
        else:
            self.condition_start_time = None
            self.event_triggered = False

    def on_pose_held_event(self):
        print("✅ 이벤트 발생: 조건 충족됨")
        self.message_box.setText("✅ 얼굴+손 위치 조건 만족 (3초 이상 유지됨)")

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
