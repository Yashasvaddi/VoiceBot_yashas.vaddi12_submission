import os
import argparse
import warnings
import speech_recognition as sr
from pydub import AudioSegment

<<<<<<< HEAD
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
=======
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore", category=UserWarning)

def convert_to_wav(input_path, output_path):
    if input_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(input_path)
        audio.export(output_path, format="wav")
    else:
        os.rename(input_path, output_path)

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError as e:
            return f"API error: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Google Speech Recognition")
    parser.add_argument("file", help="Path to the audio file (mp3 or wav)")

    args = parser.parse_args()
    input_file = args.file
    temp_wav = "temp_audio.wav"

    try:
        convert_to_wav(input_file, temp_wav)
        print("Transcribing...")
        transcription = transcribe_audio(temp_wav)
        print("\nTranscription:\n")
        print(transcription)
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
>>>>>>> 1e7e716789761d79b210689a5c264f33fe21f07e
