import pandas as pd
from response_gen import generate_response

# Load the CSV file
df = pd.read_csv("C:\\New folder\\codes\\college stuff\\VoiceBot_yashas.vaddi12_submission\\modules\\test.csv")

# Determine question column name
question_col = "Questions" if "Questions" in df.columns else "Question"

# Process each question and store results
responses = []
sources = []
confidences = []

for question in df[question_col]:
    response, source, confidence = generate_response(question)
    responses.append(response)
    sources.append(source)

# Add results to dataframe
df["Response"] = responses

# Save to CSV
output_file = "yashas.vaddi12_submissions.csv"
df.to_csv(output_file, index=False)
print(f"File saved as {output_file}")