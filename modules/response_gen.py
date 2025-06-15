import json
import boto3
import pandas as pd
import numpy as np
import faiss
import os
from langdetect import detect
from functools import lru_cache
import google.generativeai as genai
from collections import deque
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import speech_recognition as sr
import threading as th
from queue import Queue
from langdetect import DetectorFactory
DetectorFactory.seed = 0


# === Configuration ===
EMBED_FILE = "./embeddings/vectors.npy"
INDEX_FILE = "./embeddings/index.faiss"
DATASET_FILE = "./data/qa_dataset.csv"
REGION = "us-west-2"
TEXT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
CLAUDE_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
SIM_THRESHOLD = 0.50
VOICE_ID = "Aditi"
GENAI_API_KEY = "AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY"

# === Setup ===
client = boto3.client("bedrock-runtime", region_name=REGION)
polly = boto3.client("polly")
voice_input_queue = Queue()
dq = deque()
convo_history = ""

# === Load data ===
dataset = pd.read_csv(DATASET_FILE)
embeddings = np.load(EMBED_FILE)
faiss_index = faiss.read_index(INDEX_FILE)

def speak_with_polly(text):
    try:
        response = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=VOICE_ID)
        audio_stream = response["AudioStream"].read()
        audio = AudioSegment.from_file(BytesIO(audio_stream), format="mp3")
        play(audio)
    except Exception as e:
        print("Polly TTS error:", e)

@lru_cache(maxsize=128)
def embed_text(text):
    payload = json.dumps({"inputText": text})
    response = client.invoke_model(
        modelId=TEXT_EMBED_MODEL,
        body=payload,
        accept="application/json",
        contentType="application/json"
    )
    vec = np.array(json.loads(response['body'].read())["embedding"], dtype=np.float32)
    return (vec / np.linalg.norm(vec)).tolist()

def l2_to_cosine(dist):
    return 1 - (dist ** 2) / 2

@lru_cache(maxsize=128)
def query_claude(prompt, context, lang):
    global convo_history
    user_input = f"I have this information: {context}. Based on this, answer: {prompt}" if context else f"Question: {prompt}"
    content = (
        f"Content start:"
        f"Please dont mention anything I have told you between Content start and Content end they for your context and help not the part of query you need to worry about"
        f"if the input has 0 hindi letters only then respond in English. dont use hindi at all."
        f"If you want to respond in hindi, always use **Romanized Hindi** i.e Hindi written in English Letters, not in Devnagari Script. Do not use Hindi script (नमस्ते), instead use Hinglish (namaste)."
        f"You are a conversiontal chatbot reponse like a human not like a bot or llm have coneversation with the user one response at a time and be precise and short dont give unecccessary reponses"
        f"You are a female customer support executive named Lenden Mitra."
        f"Respond professionally, clearly, and empathetically. Try using modern words rather than native hindi. Use female pronouns"
        f"Content end:-"
        f"Offer actionable next steps if applicable:\n{user_input}"
        #f"Tam info -> LendenClub was established in 2014 by Mr. Bhavin Patel. The Chief Technical Officer i.e CTO is Mr. Dipesh Karki: use this info only if the qury asks about the team and Len den club as an entity"
        f"Remember this current convo is as follows: {convo_history} and give further responses accordingly. Do not highlight anything about your past interactions unless very necessary and for god's sake please dont say hello again and again"
    )
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    })
    response = client.invoke_model(
        modelId=CLAUDE_MODEL,
        body=body,
        accept="application/json",
        contentType="application/json"
    )
    output = json.loads(response['body'].read())
    for item in output.get("content", []):
        if item.get("type") == "text":
            return item["text"]
    return "I'm sorry, I couldn't generate a response."

def generate_response(query):
    lang = detect(query)
    query_vector = np.array(embed_text(query)).reshape(1, -1)
    distances, idxs = faiss_index.search(query_vector, k=1)
    dist, idx = distances[0][0], idxs[0][0]
    similarity = l2_to_cosine(dist)
    confidence = round(similarity * 100, 2)
    if similarity >= SIM_THRESHOLD:
        return dataset.iloc[idx]["Response"], "dataset", confidence
    context = dataset.iloc[idx]["Response"] if similarity > 0 else None
    return query_claude(query, context, lang), "llm", 100.0

def listen_once():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Please speak your question...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        return r.recognize_google(audio, language="en-IN")#type:ignore
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None

# === Orchestrator ===
if __name__ == "__main__":
    while True:
        question = listen_once()
        if not question:
            speak_with_polly("Sorry, I didn’t catch that. Please try again.")
            continue

        print(f" You said: {question}")

        if question.lower() == "exit":
            speak_with_polly("Goodbye!")
            break

        answer, source, score = generate_response(question)
        dq.append(answer)
        if len(dq) > 4:
            dq.popleft()

        convo_history = "".join(dq)

        print(f"🤖 ({source}, {score}%) → {answer}")
        speak_with_polly(answer)
