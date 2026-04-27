import threading
import queue
import time
import pythoncom

_q = queue.Queue()
_worker_started = False
_voice = None


def _init_voice():
    global _voice
    if _voice is not None:
        return

    import win32com.client
    _voice = win32com.client.Dispatch("SAPI.SpVoice")

    # Try to pick a female voice if available
    try:
        voices = _voice.GetVoices()
        for i in range(voices.Count):
            v = voices.Item(i)
            desc = v.GetDescription().lower()
            if "female" in desc or "zira" in desc:
                _voice.Voice = v
                break
    except Exception:
        pass


def _worker():
    pythoncom.CoInitialize()

    try:
        _init_voice()

        while True:
            text = _q.get()
            if text is None:
                break

            try:
                _voice.Speak(str(text))
            except Exception as e:
                print("SAPI TTS error:", e)

            time.sleep(0.05)

    finally:
        pythoncom.CoUninitialize()


def start_voice_worker():
    global _worker_started
    if _worker_started:
        return

    _worker_started = True
    threading.Thread(target=_worker, daemon=True).start()


def say_text(text: str):
    start_voice_worker()
    if text and isinstance(text, str):
        _q.put(text)


def say_hello():
    say_text("Hello")