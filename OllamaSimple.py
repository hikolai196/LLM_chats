from ollama import chat, generate

model = 'MODEL'

# chat
messages = [
  {
    'role': 'system',
    'content': 'You are a helpful assistant.',
  },
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    # Add user message to history
    messages.append({'role': 'user', 'content': user_input})

    # Get the model's response
    response = chat(model=model, messages=messages)

    # Extract and print the assistant's reply
    assistant_message = response['message']['content']
    print("Bot:", assistant_message)

    # Add model's response to history
    messages.append({'role': 'assistant', 'content': assistant_message})

# generate
# messages = 'Tell me a short joke.'
# response = generate(model=model, messages=messages, stream=False)
# print(response['response'])

# end with a newline
print()