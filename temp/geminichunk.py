import speech_recognition as sr
from pydub import AudioSegment
import wave
import os
import google.generativeai as genai

def translate(answer):
    genai.configure(api_key="AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY")
    model=genai.GenerativeModel("gemini-2.0-flash")
    val=model.generate_content(f"Translate this to english and also fix any grammatical errors: {answer}")
    print(f"\n\n\n\n{val.text}")

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def get_audio_duration(audio_path):
    with wave.open(audio_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def transcribe_in_chunks(audio_path, chunk_duration=5):
    recognizer = sr.Recognizer()
    full_transcript = ""

    print(f"Loading audio: {audio_path}")
    total_duration = get_audio_duration(audio_path)
    print(f"Total duration: {total_duration:.2f} seconds")
    if total_duration < chunk_duration:
        audio = recognizer.record(source, duration=chunk_duration)
        chunk_text = recognizer.recognize_google(audio)
        print(f"Chunk Transcription: {chunk_text}\n")
    else:
        with sr.AudioFile(audio_path) as source:
            for i in range(0, int(total_duration), chunk_duration):
                print(f"Processing chunk: {i}–{min(i + chunk_duration, int(total_duration))} sec")
                try:
                    audio = recognizer.record(source, duration=chunk_duration)
                    chunk_text = recognizer.recognize_google(audio)
                    print(f"Chunk Transcription: {chunk_text}\n")
                    full_transcript += chunk_text + " "
                except sr.UnknownValueError:
                    print("Could not understand this chunk.\n")
                except sr.RequestError as e:
                    print(f"API error: {e}\n")

    print("Final Transcription:")
    print(full_transcript)
    translate(full_transcript)

    with open("log.txt", "w", encoding="utf-8") as f:
        f.write(full_transcript)

if __name__ == "__main__":
    mp3_path = "C:\\New folder\\codes\\college stuff\\VoiceBot_yashas.vaddi12_submission\\modules\\test2.mp3"
    wav_path = "temp_audio.wav"

    convert_mp3_to_wav(mp3_path, wav_path)
    transcribe_in_chunks(wav_path, chunk_duration=5)
    os.remove(wav_path)
