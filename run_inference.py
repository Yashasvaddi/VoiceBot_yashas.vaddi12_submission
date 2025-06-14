import pandas as pd

# Load the CSV file
input_file = "lendenclub_100_faq.csv"
df = pd.read_csv(input_file)

# Dummy function to simulate your model's response generation
def generate_response(question):
    # Replace this with your actual model call (e.g., Claude, Bedrock, etc.)
    return f"Generated response for: {question}"

# Generate responses and add to new column
df["Model_Response"] = df["Question"].apply(generate_response)

# Save to new CSV
output_file = "lendenclub_100_faq_with_responses.csv"
df.to_csv(output_file, index=False)

print(f"File saved as {output_file}")
