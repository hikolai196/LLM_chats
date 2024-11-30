# Text completion
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage

model = OllamaLLM(model="MODELHERE")
chat_history = []

prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI named Mike, you answer questions with simple answers and no funny stuff.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

chain = prompt_template | model

def conversation():
    print("Chat, bye to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            break
    
        # response = llm.invoke(question)
        response = chain.invoke({"input": user_input, "chat_history": chat_history})
        print("AI:" + response)
        
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    conversation()


#What is fast language models