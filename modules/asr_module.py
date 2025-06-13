import os
import warnings
import speech_recognition as sr
from pydub import AudioSegment
import google.generativeai as genai

# Optional: avoid protocol buffer errors with newer protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore", category=UserWarning)

# File path (update if needed)
uploaded_file = "C:\\New folder\\codes\\college stuff\\VoiceBot_yashas.vaddi12_submission\\modules\\02268493221(02268493221)_20250503234248.mp3"

# Convert MP3 to WAV
audio = AudioSegment.from_mp3(uploaded_file)
temp_wav_path = "converted_temp.wav"
audio.export(temp_wav_path, format="wav")

# Transcribe using Google Speech Recognition
recognizer = sr.Recognizer()
with sr.AudioFile(temp_wav_path) as source:
    audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        print("Transcription Complete")
        print("Transcribed Text:\n", text)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service: {e}")

# Clean up temp file
os.remove(temp_wav_path)
