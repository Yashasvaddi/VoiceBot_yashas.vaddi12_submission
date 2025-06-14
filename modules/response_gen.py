import json
import boto3
import pandas as pd
import numpy as np
import faiss
import os
from langdetect import detect
from functools import lru_cache
import google.generativeai as genai


def translate(ans):
    genai.configure(api_key="AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY")#type:ignore
    model = genai.GenerativeModel("gemini-2.0-flash")#type:ignore


    text=model.generate_content(f"Is the given audio/text sample {ans} in hindi language and english script than only say 'HIN' if and 'ENG' if english")

    val=text.text

    print(val)
    
    return val


convo_history=""

# Directory and file paths
EMBED_FILE = "./embeddings/vectors.npy"
INDEX_FILE = "./embeddings/index.faiss"
DATASET_FILE = "./data/qa_dataset.csv"

# Constants
TEXT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
CLAUDE_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
REGION = "us-west-2"
SIM_THRESHOLD = 0.50

# AWS Client
client = boto3.client("bedrock-runtime", region_name=REGION)

# Data Loading
dataset = pd.read_csv(DATASET_FILE)
embeddings = np.load(EMBED_FILE)
faiss_index = faiss.read_index(INDEX_FILE)

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

def query_claude_cached(prompt, context, lang):
    global convo_history
    user_input = f"Question: {prompt}"
    if context:
        if translate(user_input)=="HIN":
            user_input = f"I have this information: {context}. Based on this, answer: {prompt} in hindi with english script. Never use devnagari script"
        else:
            user_input = f"I have this information: {context}. Based on this, answer: {prompt}"

    content = (
        f"You are a conversiontal chatbot reponse like a human not like a bot or llm have coneversation with the user one response at a time and be precise and short dont give unecccessary reponses"
        f"You are a customer support executive named Lenden Mitra. Respond professionally, clearly, and empathetically. "
        f"Offer actionable next steps if applicable:\n{user_input}"
        f"LendenClub was established in 2014 by Mr. Bhavin Patel. The Chief Technical Officer i.e CTO is Mr. Dipesh Karki"
        f"Remember this current convo is as follows: {convo_history} and give further responses accordingly. Do not highlight anything about your past interactions unless very necessary and for god's sake please dont say hello again and again"
    )

    messages = [{"role": "user", "content": content}]

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
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

def l2_to_cosine(dist):
    return 1 - (dist ** 2) / 2

def generate_response(query):
    language = detect(query)
    query_vector = np.array(embed_text(query)).reshape(1, -1)

    distances, idxs = faiss_index.search(query_vector, k=1)
    dist, idx = distances[0][0], idxs[0][0]

    similarity = l2_to_cosine(dist)
    confidence = round(similarity * 100, 2)

    if similarity >= SIM_THRESHOLD:
        return dataset.iloc[idx]["Response"], "dataset", confidence

    context_info = dataset.iloc[idx]["Response"] if similarity > 0 else None
    llm_output = query_claude_cached(query, context_info, language)
    source = "llm (augmented)" if context_info else "llm"
    return llm_output, source, 100.0

if __name__ == "__main__":
    while True:
        question_input = input("Enter your question: ")
        if question_input.lower() == 'exit':
            break
        answer, origin, score = generate_response(question_input)
        convo_history=convo_history+answer
        print(f"Answer ({origin}, {score}% confidence):\n{answer}")
        print("")
