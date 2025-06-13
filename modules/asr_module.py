import os
import warnings
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import google.generativeai as genai


genai.get_model('')


# Optional: avoid protocol buffer errors with newer protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

warnings.filterwarnings("ignore", category=UserWarning)

st.title("🎙️ Google Speech Recognition Transcription App")

# File upload widget
uploaded_file = st.file_uploader("Upload an MP3/WAV audio file", type=["mp3", "wav"])

if uploaded_file:
    temp_audio_path = "temp_audio.wav"

    # Save the uploaded file to a temp file and convert to wav if needed
    with open("temp_input", "wb") as f:
        f.write(uploaded_file.read())

    # Convert MP3 to WAV if needed
    if uploaded_file.name.endswith(".mp3"):
        audio = AudioSegment.from_mp3("temp_input")
        audio.export(temp_audio_path, format="wav")
    else:
        os.rename("temp_input", temp_audio_path)

    st.info("⏳ Transcribing using Google Speech Recognition...")

    # Recognize speech using Google API
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            st.success("✅ Transcription Complete")
            st.text_area("Transcribed Text:", text, height=200)
        except sr.UnknownValueError:
            st.error("😕 Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"❌ Could not request results from Google Speech Recognition service; {e}")

    # Clean up
    os.remove(temp_audio_path)
