# LenDen Mitra API

A FastAPI application for the LenDen Mitra query system that processes both single queries and CSV files.

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the API:
```
python api.py
```

The API will be available at http://localhost:8000

## API Endpoints

### 1. Process Single Query
- **URL**: `/query`
- **Method**: `POST`
- **Request Body**:
```json
{
  "text": "Your question here"
}
```
- **Response**:
```json
{
  "query": "Your question here",
  "response": "Generated response",
  "source": "dataset or llm",
  "confidence": 95.5
}
```

### 2. Process CSV File
- **URL**: `/process-csv`
- **Method**: `POST`
- **Request**: Form data with a CSV file (must contain a column named "Question" or "Questions")
- **Response**:
```json
{
  "results": [
    {
      "query": "Question 1",
      "response": "Response 1",
      "source": "dataset",
      "confidence": 98.2
    },
    ...
  ]
}
```

## Interactive Documentation

Visit http://localhost:8000/docs for interactive API documentation.