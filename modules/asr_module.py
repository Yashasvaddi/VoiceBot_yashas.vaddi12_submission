import os
import argparse
import warnings
import speech_recognition as sr
from pydub import AudioSegment

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

