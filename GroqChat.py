from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = "MODEL"

client = Groq(api_key=api_key)

messages = [{
        "role": "system",
        "content": "You are a helpful, concise, and intelligent AI assistant."
    }]

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\nGroq: Goodbye! Have a great day!")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
        )

        assistant_reply = chat_completion.choices[0].message.content.strip()

        # Append assistant's reply to message history
        messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

        print(f"Groq: {assistant_reply}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")