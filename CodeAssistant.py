from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = "MODEL"

# Initialize the model
llm = ChatOllama(
    model=model,
    temperature=0,
    max_tokens=256,
)

chat_history = [
    SystemMessage(content="You are a helpful software developer specialized in Python.")
]

while True:
    user_input = input("User: ")
    
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting the chat.")
        break

    # Add user's message to history
    chat_history.append(HumanMessage(content=user_input))

    # Get response from model
    response = llm.invoke(chat_history)

    # Print and store the assistant's reply
    print("Assistant:", response.content)
    chat_history.append(AIMessage(content=response.content))