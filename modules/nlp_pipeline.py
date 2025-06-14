import boto3
import pandas as pd
import numpy as np
import faiss
import json
import os
from tqdm import tqdm

MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-west-2"
# DATASET_PATH = "./data/qa_dataset.csv"
# OUTPUT_DIR = "./embeddings"
# EMBED_FILE = os.path.join(OUTPUT_DIR, "vectors.npy")
# INDEX_FILE = os.path.join(OUTPUT_DIR, "index.faiss")

# os.makedirs(OUTPUT_DIR, exist_ok=True)


EMBED_FILE = "C:\\New folder\\codes\\college stuff\\VoiceBot_yashas.vaddi12_submission\\embeddings\\vectors.npy"
INDEX_FILE = "C:\\New folder\\codes\\college stuff\\VoiceBot_yashas.vaddi12_submission\\embeddings\\index.faiss"
DATASET_PATH = "C:\\New folder\\codes\college stuff\\VoiceBot_yashas.vaddi12_submission\\data\\qa_dataset.csv"

client = boto3.client("bedrock-runtime", region_name=REGION)

def fetch_embedding(text):
    payload = json.dumps({"inputText": text})
    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=payload,
        accept="application/json",
        contentType="application/json"
    )
    vec = np.array(json.loads(resp['body'].read())["embedding"], dtype=np.float32)
    return vec / np.linalg.norm(vec)

def build_index(questions):
    vectors = np.array([fetch_embedding(q) for q in tqdm(questions, desc="Embedding Progress")], dtype=np.float32)
    np.save(EMBED_FILE, vectors)

    dims = vectors.shape[1]
    idx = faiss.IndexFlatL2(dims)
    idx.add(vectors) #type:ignore
    faiss.write_index(idx, INDEX_FILE)

    print("Index and embeddings stored successfully.")

if __name__ == "__main__":
    dataset = pd.read_csv(DATASET_PATH)
    build_index(dataset["Question"].tolist())
