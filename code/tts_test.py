#pip install gtts
from gtts import gTTS
import os

text = "안녕하세요! 저희는 그로밋입니다."
tts = gTTS(text=text, lang='ko')
tts.save("output3.mp3")
os.system("start output3.mp3") 
