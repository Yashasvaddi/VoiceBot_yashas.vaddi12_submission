from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import io
import numpy as np
from modules.response_gen import generate_response
import uvicorn

app = FastAPI(title="LenDen Mitra API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:3000"] for stricter settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.post("/query")
async def process_query(query: Query):
    response, source, confidence = generate_response(query.text)
    return {
        "query": query.text,
        "response": response,
        "source": source,
        "confidence": convert_numpy_types(confidence)
    }

@app.post("/process-csv")
async def process_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    question_col = next((col for col in df.columns if col.lower() in ["question", "questions"]), None)
    if not question_col:
        return JSONResponse(
            status_code=400,
            content={"error": "CSV must contain a column named 'Question' or 'Questions'"}
        )
    
    results = []
    for question in df[question_col]:
        response, source, confidence = generate_response(question)
        results.append({
            "query": str(question),  # Ensure it's a string
            "response": response,
            "source": source,
            "confidence": convert_numpy_types(confidence)
        })
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)