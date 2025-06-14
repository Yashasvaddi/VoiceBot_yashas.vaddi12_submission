import pandas as pd
from modules.response_gen import generate_response

# Load the CSV file
df = pd.read_csv("lendenclub_100_faq.csv")

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
    confidences.append(confidence)

# Add results to dataframe
df["Model_Response"] = responses
df["Response_Source"] = sources
df["Confidence"] = confidences

# Save to CSV
output_file = "lendenclub_100_faq_with_responses.csv"
df.to_csv(output_file, index=False)
print(f"File saved as {output_file}")