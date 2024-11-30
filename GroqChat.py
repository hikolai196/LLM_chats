import os

from groq import Groq

os.environ['GROQ_API_KEY'] = "GROQAPIHERE"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain what is fast language models",
        }
    ],
    model="llama3-70b-8192",
)

print(chat_completion.choices[0].message.content)