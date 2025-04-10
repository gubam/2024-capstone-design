import sys
sys.coinit_flags = 2  # STA 모드 강제 설정 (추가)

# 기존 import 밑에 추가
from predict import predict
from predict_bilstm import predict


import os
import cv2
import numpy as np
import sounddevice as sd
import wavio
import time
import subprocess
from pywinauto import application
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt

import pythoncom
pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)  # STA 모드 강제 설정

import warnings
warnings.simplefilter("ignore", UserWarning)  # UserWarning 전체 무시
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Deprecated 경고 무시


# 경로 설정
AUDIO_SAVE_PATH = r"C:/Users/SAMSUNG/OneDrive/바탕 화면/Coding/캡스톤/project/recoding"
VIDEO_SAVE_PATH = r"C:/Users/SAMSUNG/OneDrive/바탕 화면/Coding/캡스톤/project/video"

os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translation System")
        self.setGeometry(100, 100, 1400, 800)

        # 웹캠 화면
        self.label_webcam = QLabel(self)
        self.label_webcam.setGeometry(20, 20, 640, 480)

        # 얼굴 인식 불가 메시지
        self.label_warning = QLabel("얼굴을 인식할 수 없습니다.", self)
        self.label_warning.setGeometry(20, 500, 300, 30)
        self.label_warning.setStyleSheet("color: red; font-size: 14px;")
        self.label_warning.hide()  # 처음엔 숨김

        # 오른쪽 큰 사각형
        self.right_panel = QLabel(self)
        self.right_panel.setGeometry(680, 20, 640, 480)
        self.right_panel.setStyleSheet("background-color: #E0E0E0;")

        # 버튼 설정
        self.btn_kor_to_sign = QPushButton("한글 → 수어\n음성녹음", self)
        self.btn_kor_to_sign.setGeometry(20, 570, 310, 120)
        self.btn_kor_to_sign.setStyleSheet("background-color: #96A5FF; color: white; font-size: 16px;")
        self.btn_kor_to_sign.clicked.connect(self.toggle_audio_recording)

        self.btn_sign_to_kor = QPushButton("수어 → 한글\n영상녹화", self)
        self.btn_sign_to_kor.setGeometry(350, 570, 310, 120)
        self.btn_sign_to_kor.setStyleSheet("background-color: #96A5FF; color: white; font-size: 16px;")
        self.btn_sign_to_kor.clicked.connect(self.toggle_video_recording)

        # 얼굴 인식 설정
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.is_recording_audio = False
        self.is_recording_video = False
        self.audio_data = []
        self.video_writer = None
        self.stream = None
        self.last_in_box_time = 0

        # 타이머 라벨
        self.label_timer = QLabel("00:00", self)
        self.label_timer.setGeometry(350, 700, 310, 30)
        self.label_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_timer.setStyleSheet("color: black; font-size: 18px;")
        self.label_timer.hide()

        # 타이머 객체 생성
        self.timer_timer = QTimer()
        self.timer_timer.timeout.connect(self.update_timer)
        self.record_start_time = None


    def update_timer(self):
        if self.record_start_time:
            elapsed = int(time.time() - self.record_start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self.label_timer.setText(f"{minutes:02d}:{seconds:02d}")


    # 실시간 카메라 업데이트 및 얼굴 인식
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # 고정된 빨간색 사각형 설정
        fixed_rect_w, fixed_rect_h = 200, 200
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fixed_rect_x = frame_width // 2 - fixed_rect_w // 2
        fixed_rect_y = frame_height // 2 - fixed_rect_h // 2 - 50

        # 버퍼 영역
        buffer = 50
        buffered_rect_x = fixed_rect_x - buffer
        buffered_rect_y = fixed_rect_y - buffer
        buffered_rect_w = fixed_rect_w + buffer * 2
        buffered_rect_h = fixed_rect_h + buffer * 2

        # 가장 가까운 얼굴 찾기
        min_distance = float('inf')
        closest_face = None

        for (x, y, w, h) in faces:
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # 사각형 중심과 얼굴 중심 간 거리 계산
            rect_center_x = fixed_rect_x + fixed_rect_w // 2
            rect_center_y = fixed_rect_y + fixed_rect_h // 2
            distance = ((rect_center_x - face_center_x) ** 2 + (rect_center_y - face_center_y) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_face = (x, y, w, h)

         # 얼굴 인식 및 사각형 전환
        face_in_box = False
        current_time = time.time()
        for (x, y, w, h) in faces:
            # 얼굴의 일부가 탐지 영역에 닿기만 해도 초록색
            if (buffered_rect_x < x + w and x < buffered_rect_x + buffered_rect_w) and \
               (buffered_rect_y < y + h and y < buffered_rect_y + buffered_rect_h):
                face_in_box = True
                self.last_in_box_time = current_time
                break

        if closest_face:
            x, y, w, h = closest_face
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # 버퍼 영역 안에 있는지 확인
            if (buffered_rect_x < face_center_x < buffered_rect_x + buffered_rect_w) and (buffered_rect_y < face_center_y < buffered_rect_y + buffered_rect_h):
                face_in_box = True
                self.last_in_box_time = current_time

         # 사각형 색상 및 경고 메시지 처리
        if face_in_box or (current_time - self.last_in_box_time <= 1.0):
            color = (0, 255, 0)  # 초록색
            self.label_warning.hide()
            self.btn_sign_to_kor.setEnabled(True)
        else:
            color = (0, 0, 255)  # 빨간색
            self.label_warning.show()
            self.btn_sign_to_kor.setEnabled(False)

        cv2.rectangle(frame, (fixed_rect_x, fixed_rect_y), (fixed_rect_x + fixed_rect_w, fixed_rect_y + fixed_rect_h), color, 2)

        if self.is_recording_video and self.video_writer:
            self.video_writer.write(frame)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label_webcam.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

    # def run_dollars_mono(self):
    #     dollars_mono_path = r"C:/Users/SAMSUNG/Downloads/Dollars_MONO_250103_2/Dollars_MONO.exe"
    #     avi_file_path = os.path.join(VIDEO_SAVE_PATH, "recording.avi")

    #     if os.path.exists(dollars_mono_path) and os.path.exists(avi_file_path):
    #         try:
    #             # 프로그램 실행
    #             subprocess.Popen([dollars_mono_path, avi_file_path])
    #             time.sleep(2)  # 실행 후 로딩 시간 대기

    #             # pywinauto로 프로그램 컨트롤
    #             app = application.Application().connect(title="Dollars_MONO")
    #             window = app.window(title="Dollars_MONO")

    #             # 이메일과 비밀번호 입력
    #             window['Edit'].type_keys("hedgehog7250@gmail.com")
    #             window['Edit2'].type_keys("johnshopkins25!")  # 비밀번호 입력
    #             time.sleep(0.5)  # 입력 대기

    #             # 로그인 버튼 클릭
    #             window['Login'].click()
    #             print("Dollars_MONO 자동 로그인 완료!")
    #         except Exception as e:
    #             print(f"Dollars_MONO 실행 오류: {e}")
    #     else:
    #         print("Dollars_MONO 실행 파일이나 AVI 파일이 없습니다.")

    # 음성 녹음 시작 및 저장
    def toggle_audio_recording(self):
        if not self.is_recording_audio:
            self.is_recording_audio = True
            self.btn_kor_to_sign.setStyleSheet("background-color: #FF7E9D; color: white; font-size: 16px;")
            self.audio_data = []
            self.start_recording()
        else:
            self.is_recording_audio = False
            self.btn_kor_to_sign.setStyleSheet("background-color: #96A5FF; color: white; font-size: 16px;")
            self.stop_and_save_audio()
            
    # 음성 녹음
    def start_recording(self):
        try:
            self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=44100)
            self.stream.start()
        except Exception as e:
            print(f"녹음 시작 오류: {e}")

    def audio_callback(self, indata, frames, time, status):
        if self.is_recording_audio:
            self.audio_data.append(indata.copy())

    def stop_and_save_audio(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            filename = os.path.join(AUDIO_SAVE_PATH, "recording.wav")
            if os.path.exists(filename):
                os.remove(filename)
            wavio.write(filename, audio_array, 44100, sampwidth=2)
            print(f"녹음이 저장되었습니다: {filename}")

    # 영상 녹화
    def toggle_video_recording(self):
        if not self.is_recording_video:
            self.is_recording_video = True
            self.btn_sign_to_kor.setStyleSheet("background-color: #FF7E9D; color: white; font-size: 16px;")
            self.video_filename = os.path.join(VIDEO_SAVE_PATH, "recording.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 20.0, (640, 480))

            # ✅ 타이머 시작
            self.record_start_time = time.time()
            self.label_timer.show()
            self.timer_timer.start(1000)  # 1초마다 update_timer 실행

        else:
            self.is_recording_video = False
            self.btn_sign_to_kor.setStyleSheet("background-color: #96A5FF; color: white; font-size: 16px;")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                print(f"녹화가 저장되었습니다: {self.video_filename}")

                # ✅ 타이머 종료
                self.label_timer.hide()
                self.timer_timer.stop()

                # ✅ 녹화된 영상으로 바로 예측
                result = predict(self.video_filename)
                print("예측된 단어:", result)

                # ✅ 오른쪽 사각형에 결과 출력
                self.right_panel.setText(result)
                self.right_panel.setStyleSheet("font-size: 24px; color: black; background-color: #E0E0E0;")



    def closeEvent(self, event):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.video_writer:
            self.video_writer.release()
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec())
