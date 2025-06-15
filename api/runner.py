import requests

API_URL = "http://localhost:8000"  # Change to your deployed URL if needed

# === 1. Single query
def send_single_query(text):
    response = requests.post(f"{API_URL}/query", json={"text": text})
    if response.status_code == 200:
        data = response.json()
        print("📤 Query:", data["query"])
        print("🤖 Response:", data["response"])
        print("🧠 Source:", data["source"])
        print("🎯 Confidence:", data["confidence"])
    else:
        print("❌ Failed:", response.text)

# === 2. CSV upload
def send_csv(file_path):
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/csv")}
        response = requests.post(f"{API_URL}/process-csv", files=files)

    if response.status_code == 200:
        results = response.json().get("results", [])
        print(f"✅ {len(results)} queries processed:")
        for item in results:
            print(f"- 📤 {item['query']}")
            print(f"  🤖 {item['response']} (via {item['source']}, {item['confidence']}%)\n")
    else:
        print("❌ CSV Upload Failed:", response.text)

# === Run it
if __name__ == "__main__":
    # Example usage
    #send_single_query("What is LenDen Club?")
    send_csv("lendenclub_100_faq.csv")  # Uncomment to test CSV upload
