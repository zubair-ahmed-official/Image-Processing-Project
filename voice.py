import pyttsx3
import time

engine = pyttsx3.init()

# 🔹 Set speaking speed
engine.setProperty("rate", 160)

# 🔹 Select female voice
voices = engine.getProperty("voices")

for voice in voices:
    # Many Windows female voices contain "female" or specific names like "Zira"
    if "female" in voice.name.lower() or "zira" in voice.name.lower():
        engine.setProperty("voice", voice.id)
        break

last_spoken_time = 0
COOLDOWN = 4  # seconds

def say_hello():
    global last_spoken_time
    now = time.time()

    if now - last_spoken_time > COOLDOWN:
        engine.say("Hello")
        engine.runAndWait()
        last_spoken_time = now
