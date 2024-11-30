# Import necessary libraries
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
llm = ChatOllama(
    model="MODELHERE",
    temperature=0.1,
)

prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful software developer specialized in Python."),
        ("human", "{input}"),
    ])

chain = prompt | llm
ai_msg = chain.invoke({
        "input": "Create a snake game. Use best practices and include code comments.",
    })

print(ai_msg.content)