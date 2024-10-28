import os
import json
from openai import AzureOpenAI
import requests

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-08-01-preview"
)

# Define the deployment you want to use for your chat completions API calls

deployment_name = "gpt-4"

def send_email_via_function(to_email, subject, body):
    """Send an email by calling the Azure Function."""
    function_url = os.getenv("AZURE_FUNCTION_URL")
    
    # Prepare the payload for the Azure Function
    payload = {
        "to_email": to_email,
        "subject": subject,
        "body": body
    }
    
    # Make the POST request to the Azure Function
    response = requests.post(function_url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json()}


def run_conversation():
    # Initial user message
    messages = [{"role": "user", "content": "Send an email to Min.Cheng@knightec.se with the subject 'Greetings' and body 'Hello, this is a test email!'"}] # Single function call
    #messages = [{"role": "user", "content": "What's the current time in San Francisco, Tokyo, and Paris?"}] # Parallel function call with a single tool/function defined

    # Define the function for the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Sends an email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_email": {"type": "string", "description": "The recipient's email address"},
                        "subject": {"type": "string", "description": "The subject of the email"},
                        "body": {"type": "string", "description": "The body of the email"}
                    },
                "required": ["to_email", "subject", "body"]
        }
            }
        },
    ]

    # First API call: Ask the model to use the function
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Process the model's response
    response_message = response.choices[0].message
    messages.append(response_message)

    print("Model's response:")  
    print(response_message)  

    # Handle function calls
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"Function call: {function_name}")
                    print(f"Function arguments: {function_args}")

                    if function_name == "send_email":
                        # Call the Azure Function to send an email
                        function_response = send_email_via_function(
                            to_email=function_args.get("to_email"),
                            subject=function_args.get("subject"),
                            body=function_args.get("body")
                        )
                    else:
                        function_response = json.dumps({"error": "Unknown function"})

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response),
                    })
    else:
        print("No tool calls were made by the model.")
    # Second API call: Get the final response from the model
    final_response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
    )

    return final_response.choices[0].message.content

# Run the conversation and print the result
print(run_conversation())



