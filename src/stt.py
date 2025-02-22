import pyaudio
import wave
from faster_whisper import WhisperModel
from transformers import pipeline

# Load NLP Model
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1") #use "facebook/bart-large-mnli" once implemented fully

# Define possible commands
COMMAND_LABELS = [
    "check weather", "weather forecast",
    "tell time", "what time is it", "current time",
    "tell a joke", "say something funny",
    "open YouTube", "play music", "start music",
    "search Google", "Google search"
]

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
AUDIO_FILE = "speech.wav"

def record_audio():
    """Records audio from the microphone and saves it as a WAV file."""
    audio = pyaudio.PyAudio()
    
    print("\n🎤 Speak now...")

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("✅ Recording complete!")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save the recorded audio as a WAV file
    with wave.open(AUDIO_FILE, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    print(f"🎙️ Audio saved as {AUDIO_FILE}")

def transcribe_audio():
    # transcribes the recorded audio file
    print("\n Transcribing audio...")

    model = WhisperModel("base", compute_type="int8")

    segments, _ = model.transcribe("speech.wav", language="en")

    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    if transcribed_text.strip():
        print("🗣️ Transcription:", transcribed_text.strip())
    else:
        print(" No transcription detected. Try speaking louder.")

    return transcribed_text.strip()

def understand_command(text):
    """Processes the transcribed text and identifies the user's intent."""
    
    result = classifier(text, COMMAND_LABELS)
    
    # Get the highest confidence label
    intent = result["labels"][0]
    confidence = result["scores"][0]

    print(f"🧠 Detected Intent: {intent} (Confidence: {confidence:.2f})")
    
    return intent if confidence >= 0.7 else None  # Only return intent if confidence is high
       


def execute_command(intent):
    # executes the action based on the detected intent
    
    if intent in ["tell time", "what time is it", "current time"]:
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M")
        print(f"⏰ The time is {current_time}.")
    
    elif intent in ["tell a joke", "say something funny"]:
        print("Why was the broom late for work? .... it over-swept! 😂")

    elif intent in ["check weather", "weather forecast"]:
        print("🌤️ Sorry, not that advanced yet! Hahaha, coming soooon!")

    elif intent in ["open YouTube"]:
        import webbrowser
        webbrowser.open("https://www.youtube.com")
        print("🎬 Opening YouTube...")

    elif intent in ["play music", "start music"]:
        print("🎵 Playing some music... (Feature coming soon!)")

    elif intent in ["search Google", "Google search"]:
        print("🔎 What do you want to search for? (Feature coming soon!)")

    else:
        print("🤖 I didn't understand. Can you repeat?")

if __name__ == "__main__":
    record_audio()
    text = transcribe_audio()
    
    if text:
        intent = understand_command(text)
        if intent:
            execute_command(intent)
        else:
            print("I didn't understand. Please try again.")
