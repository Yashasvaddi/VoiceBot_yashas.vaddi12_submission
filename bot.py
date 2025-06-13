import boto3
import json

# Create a Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'  # Choose your preferred region where Bedrock is available
)

# Define your prompt
prompt = "Explain quantum computing in simple terms."

# Prepare request for Claude model
request_body = {
    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
    "max_tokens_to_sample": 300,
    "temperature": 0.5,
    "top_p": 0.9,
}

# Invoke the model
response = bedrock_runtime.invoke_model(
    modelId="anthropic.claude-v2",  # Choose your preferred model
    body=json.dumps(request_body)
)

# Parse and print the response
response_body = json.loads(response.get('body').read())
print(response_body.get('completion'))