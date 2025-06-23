import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("API_VERSION")
 
if not all([endpoint, deployment, subscription_key, api_version]):
    raise ValueError("Missing required environment variables. Check your .env file.")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version=api_version,
)

system_message = {
    "role": "system",
    "content": "You are an AI assistant that helps people find information."
}

chat_prompt = [system_message]

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting the chat. Goodbye!")
        break
    
    chat_prompt.append({
            "role": "user",
            "content": user_input
        })
    
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=chat_prompt,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        
        ai_message = completion.choices[0].message.content
        chat_prompt.append({
            "role": "assistant",
            "content": ai_message
        })
        print(f"assistant: {ai_message}\n")
    
    except Exception as e:
        print(f"An error occurred: {e}")