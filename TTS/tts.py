from gtts import gTTS
import os


def speak(text, lang="ko", speed=False):
    tts = gTTS(text=text, lang=lang, slow=speed)
    tts.save("./tts.mp3")  # tts.mp3로 저장
    os.system("afplay " + "./tts.mp3")  # 말하기
