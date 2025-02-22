import pvporcupine          # detects wake word Hey Clair
import pyaudio              # captures real-time microphone input
import struct               # converts raw audio data into an understandable format for pvpircupine
import numpy as np          # helps with audio processing
import os                   # to handle files path


# # Initialize wake word detection
# porcupine = pvporcupine.create(
#     access_key="#",
#     keyword_paths=[os.path.join(os.path.dirname(__file__), "hey_clair.ppn")] #to change ASAP when get the file and permission from CONSOLE PORCUPINE
# )

# JUST TO TEST IN THE MEANTIME THAT PROCUPINE ACCEPT MY REQUEST:
porcupine = pvporcupine.create(
    access_key="YOUR_PICOVOICE_ACCESS_KEY",
    keyword="picovoice" 
)


# set up microphone audio input
# microphone open for continuous audio input.
# using Porcupine’s sample rate & frame length for accurate detection.
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=porcupine.sample_rate,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

# main loop - listen for hey clair!
print("Listening for 'Hey Clair!'...")

try:
    while True:
        # read audio data
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
        pcm_numpy = np.array(pcm_unpacked, dtype=np.int16)
        
        # check if is awake word detected
        keyword_index = porcupine.process(pcm_numpy)
        if keyword_index >= 0:
            print("Wake word detected!")
            break # stop listening, trigger STT
        
except KeyboardInterrupt:
    print("Stopping wake word detection...")
finally:
    # clean up resources
    stream.stop_stream()
    stream.close()
    audio.terminate()
    porcupine.delete()