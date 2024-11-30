from ollama import chat, generate

messages = [
  {
    'role': 'user',
    'content': 'Tell me a short joke.',
  },
]

# chat
# response = chat('MODELHERE', messages=messages)
# print(response['message']['content'])

# stream chat
# for part in chat('MODELHERE', messages=messages, stream=True):
#     print(part['message']['content'], end='', flush=True)

# generate
response = generate('MODELHERE', 'Tell me a short joke.')
print(response['response'])

# end with a newline
print()