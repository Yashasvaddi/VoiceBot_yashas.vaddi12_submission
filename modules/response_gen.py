import boto3
import json

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

# Claude requires this format: a "prompt" with Human/Assistant dialogue
prompt = """Human: who is the president of india.
Assistant:"""

# Claude-compatible payload
body = {
    "prompt": prompt,
    "max_tokens_to_sample": 300,
    "temperature": 0.7,
    "stop_sequences": ["\nHuman:"]
}

# Make the call to Bedrock
response = bedrock.invoke_model(
    modelId="anthropic.claude-instant-v1",
    body=json.dumps(body),
    contentType="application/json",
    accept="application/json"
)

# Parse and print the response
response_body = json.loads(response['body'].read())
print("Claude's Response:\n", response_body['completion'])
