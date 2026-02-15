import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty("rate", 160)

# choose female voice if available (Windows often has Zira)
voices = engine.getProperty("voices")
for v in voices:
    if "zira" in v.name.lower() or "female" in v.name.lower():
        engine.setProperty("voice", v.id)
        break

_last_spoken = 0
COOLDOWN = 4  # seconds

def say_hello():
    global _last_spoken
    now = time.time()
    if now - _last_spoken >= COOLDOWN:
        engine.say("Hello")
        engine.runAndWait()
        _last_spoken = now
