import sys
import cv2
import threading
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit, QHBoxLayout,
    QVBoxLayout, QFileDialog
)
from PyQt6.QtCore import Qt
import time
import os
from modules.sign2audio import audio_to_sign, sign_to_audio
import sounddevice as sd
from scipy.io.wavfile import write as write_wav
from modules.motion_generation import render_multiple_folders
from PyQt6.QtGui import QFont
import pygame

class StreamRedirect:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, msg):
        msg = msg.strip()
        if msg:
            self.text_widget.append(msg)

    def flush(self):
        pass
def play_mp3():
    pygame.mixer.init()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "file", "tts_output.mp3")

    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

    try:
        pygame.mixer.music.load(MODEL_PATH)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()
        pygame.mixer.quit()  # 🔴 여기서 리소스를 완전히 해제
    except Exception as e:
        print(f"MP3 재생 오류: {e}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("그로밋의 수어 번역기")
        self.resize(900, 500)

        # 버튼 및 메시지 박스
        font = QFont()
        font.setPointSize(16)
        fonts = QFont()
        fonts.setPointSize(10)     
        self.record_button = QPushButton("웹캠 녹화 시작\n(수어 -> 음성)")
        self.audio_button = QPushButton("녹음\n(음성 -> 수어)")  # UI용
        self.record_button.setFont(font)
        self.audio_button.setFont(font)

        self.log_box = QTextEdit()
        self.log_box.setPlaceholderText("터미널 출력 로그")
        self.log_box.setReadOnly(True)
        self.log_box.setFont(fonts)

        self.message_box = QTextEdit()
        self.message_box.setFont(fonts)
        self.message_box.setPlaceholderText("메시지 박스")

        self.is_recording_audio = False
        self.audio_thread = None
        self.record_button.setFixedHeight(100)
        self.audio_button.setFixedHeight(100)
        
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.message_box,stretch=2)
        bottom_layout.addWidget(self.log_box,stretch=1)

        # 레이아웃
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.record_button)
        h_layout.addWidget(self.audio_button)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addLayout(bottom_layout)  
        self.setLayout(v_layout)

        # 상태 변수
        self.is_recording = False
        self.record_thread = None

        # 버튼 이벤트
        self.record_button.clicked.connect(self.toggle_recording)
        self.audio_button.clicked.connect(self.toggle_audio_recording)
        sys.stdout = StreamRedirect(self.log_box)
        sys.stderr = StreamRedirect(self.log_box)


    def record_video(self):
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 60)
            # self.message_box.clear()
            if not cap.isOpened():
                self.message_box.append("웹캠을 열 수 없습니다.")
                return

            # 저장 경로 선택
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(BASE_DIR, "file", "input.mp4")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if actual_fps == 0:
                actual_fps = 30.0

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(file_path, fourcc, actual_fps, (width, height))

            win_name = '녹화 중 - q로 종료'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 1280, 720)
            cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
            cv2.moveWindow(win_name, 100, 100)

            # ✅ 3초 카운트다운
            start_time = time.time()
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if not ret:
                    break
                countdown = 3 - int(time.time() - start_time)
                preview = cv2.resize(frame, (1280, 720))
                cv2.putText(preview, f"{countdown} second", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.imshow(win_name, preview)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            cv2.putText(preview, f"recordingq", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            print("녹화 시작")
            self.message_box.append("녹화 시작")

            # ✅ 본격적인 녹화 루프
            while self.is_recording:
                ret, frame = cap.read()
                if not ret:
                    break

                out.write(frame)  # 원본 프레임을 저장

                # 👉 미리보기용 복사본에 텍스트 추가
                preview = cv2.resize(frame.copy(), (1280, 720))
                cv2.putText(preview, "Recording...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                cv2.imshow(win_name, preview)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print("녹화 종료")
            self.message_box.append("녹화 종료됨")
            self.message_box.append("●영상 분석중●")

            out = sign_to_audio(file_path)

            self.message_box.append(f"문장 출력 : {out}")
            play_mp3()
            self.message_box.append(f"음성 출력중")
            self.message_box.append(f"완료")
            self.message_box.append("---------")




        except KeyboardInterrupt:
            print("Ctrl+C - 녹화 중단됨")
            self.is_recording = False
        finally:
            self.is_recording = False
            self.record_button.setText("웹캠 녹화 시작")
        
    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_button.setText("녹화 중... (Ctrl+C로 중단)")
            self.message_box.append("웹캠 녹화 시작")

            # 스레드로 녹화 시작
            self.record_thread = threading.Thread(target=self.record_video)
            self.record_thread.start()
        else:
            self.message_box.append("이미 녹화 중입니다.")

    def toggle_audio_recording(self):
        if not self.is_recording_audio:
            self.is_recording_audio = True
            self.audio_button.setText("녹음 중지")
            self.message_box.append("🎙️ 오디오 녹음 시작")
            print("🎙️ 오디오 녹음 시작")
            self.audio_thread = threading.Thread(target=self.record_audio)
            self.audio_thread.start()
        else:
            self.is_recording_audio = False
            self.audio_button.setText("녹음")
            print("🛑 녹음 중지 요청됨")

    
    def record_audio(self):
        try:
            # self.message_box.clear()

            sample_rate = 44100
            channels = 1
            print("🔴 녹음 중... 정지 버튼을 누르세요.")

            max_duration = 300  # 최대 녹음 길이 (예비용)
            start_time = time.time()

            # 🔴 녹음 시작
            audio = sd.rec(int(max_duration * sample_rate), samplerate=sample_rate,
                        channels=channels, dtype='int16')

            while self.is_recording_audio:
                sd.sleep(100)

            # ⏹️ 정지 시점
            sd.stop()
            end_time = time.time()
            actual_duration = end_time - start_time
            actual_samples = int(actual_duration * sample_rate)

            print(f"🟢 녹음 완료 - 실제 길이: {actual_duration:.2f}초")

            # 🔄 슬라이싱: 실제 녹음된 부분만 저장
            trimmed_audio = audio[:actual_samples]

            # 저장 경로
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            audio_path = os.path.join(BASE_DIR, "file", "stt_input.mp3")

            if os.path.exists(audio_path):
                os.remove(audio_path)
            write_wav(audio_path, sample_rate, trimmed_audio)

            self.message_box.append("🎧 오디오 저장 완료")
            gloss = audio_to_sign()
            print(gloss)
            self.message_box.append(f"추출 단어 : {gloss}")

            # 🎬 영상 재생
            render_multiple_folders(gloss)

        except Exception as e:
            print(f"오디오 녹음 오류: {e}")
            self.message_box.append(f"오디오 녹음 오류: {e}")
        finally:
            self.is_recording_audio = False
            self.audio_button.setText("녹음")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("앱 종료")
