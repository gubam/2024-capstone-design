from modules.sign2audio import sign_to_audio
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "file", "input.mp4")
sign_to_audio(file_path)