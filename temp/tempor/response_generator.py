import json
import boto3
import pandas as pd
import numpy as np
import faiss
import os
from langdetect import detect

# Paths
EMBEDDINGS_FILE = "C:\\New folder\\codes\\college stuff\\tempor\\data\\qa_embeddings.npy"
FAISS_INDEX_FILE = "C:\\New folder\\codes\\college stuff\\tempor\\data\\qa_faiss_index.bin"
CSV_PATH = "C:\\New folder\\codes\\college stuff\\tempor\\data\\qa_dataset.csv"

# Constants
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_REGION = "us-west-2"
SIMILARITY_THRESHOLD = 0.50

# Clients
bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# Load artifacts
df = pd.read_csv(CSV_PATH)
qa_embeddings = np.load(EMBEDDINGS_FILE)
index = faiss.read_index(FAISS_INDEX_FILE)

def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        body=body,
        modelId=EMBEDDING_MODEL_ID,
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    vec = np.array(result['embedding'], dtype=np.float32)
    return vec / np.linalg.norm(vec)

def cosine_similarity_from_l2(d):
    return 1 - (d**2) / 2

def get_claude_response(question, context=None, lang='en'):
    user_prompt = f"Question: {question}"
    if context:
        user_prompt = f"I have this information: {context}. Based on this, answer: {question}"

    if lang == 'hi':
        content = f"आप एक ग्राहक सेवा सहायक हैं। कृपया प्रश्न का उत्तर हिंदी में दें:\n{user_prompt}\n\nउत्तर:"
    else:
        content = f"You are a customer support executive. Respond professionally and clearly:\n{user_prompt}"

    messages = [{"role": "user", "content": content}]
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9
    })

    response = bedrock.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=body,
        accept="application/json",
        contentType="application/json"
    )

    response_data = json.loads(response['body'].read())
    for block in response_data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return "I'm sorry, I couldn't generate a response."

def get_response(question):
    lang = detect(question)
    query_vec = get_embedding(question).reshape(1, -1)
    distances, indices = index.search(query_vec, k=1)

    best_distance = distances[0][0]
    best_idx = indices[0][0]
    similarity = cosine_similarity_from_l2(best_distance)
    score = round(similarity * 100, 2)

    if similarity >= SIMILARITY_THRESHOLD:
        return df.iloc[best_idx]['Response'], 'dataset', score

    context = df.iloc[best_idx]['Response'] if similarity > 0 else None
    llm_answer = get_claude_response(question, context, lang)
    return llm_answer, 'llm (augmented)' if context else 'llm', 100.0
