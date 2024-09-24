import speech_recognition as sr
import time
from tts import speak

try:
    while True:
        r = sr.Recognizer()

        with sr.Microphone() as source:
            print("음성을 입력하세요.")
            start_time = time.time()
            audio = r.listen(source)
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"{elapsed_time:.2f} 초가 지났습니다.")
            try:
                google_result = r.recognize_google(audio, language="ko-KR")
                whisper_result = r.recognize_whisper(audio, language="ko")
                print("구글 : " + google_result)
                print("위스퍼 : " + whisper_result)
                if "아영" in google_result:
                    speak("네, 주문을 도와드리겠습니다.")
            except sr.UnknownValueError:
                print("오디오를 이해할 수 없습니다.")
            except sr.RequestError as e:
                print(f"에러가 발생하였습니다. 에러원인 : {e}")

except KeyboardInterrupt:
    pass
