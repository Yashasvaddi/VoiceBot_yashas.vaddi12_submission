import boto3
import pandas as pd
import numpy as np
import faiss
import json
import os

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
BEDROCK_REGION = "us-west-2"
CSV_PATH = "C:\\New folder\\codes\\college stuff\\tempor\\data\\qa_dataset.csv"
EMBEDDINGS_FILE = "C:\\New folder\\codes\\college stuff\\tempor\\data\\qa_embeddings.npy"
FAISS_INDEX_FILE = "C:\\New folder\\codes\\college stuff\\tempor\\data\\qa_faiss_index.bin"

# AWS Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=body,
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response['body'].read())
    vec = np.array(result['embedding'], dtype=np.float32)
    return vec / np.linalg.norm(vec)

# Load QA dataset
df = pd.read_csv(CSV_PATH)
questions = df['Question'].tolist()

# Generate and normalize embeddings
embeddings = np.array([get_embedding(q) for q in questions], dtype=np.float32)

# Save the embeddings
np.save(EMBEDDINGS_FILE, embeddings)

# Create and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_FILE)

print("FAISS index and embeddings saved!")
