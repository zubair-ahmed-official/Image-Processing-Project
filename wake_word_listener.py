import speech_recognition as sr
import threading

wake_word_detected = False


def listen_for_wake_word():
    global wake_word_detected

    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                audio = r.listen(source, phrase_time_limit=3)

            text = r.recognize_google(audio).lower()

            if "hi matilda" in text:
                wake_word_detected = True
                print("Wake word detected!")

        except:
            pass


def start_wake_listener():
    thread = threading.Thread(target=listen_for_wake_word, daemon=True)
    thread.start()


def is_wake_word_detected():
    global wake_word_detected
    if wake_word_detected:
        wake_word_detected = False
        return True
    return False