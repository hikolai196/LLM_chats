import os
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# The model path
model_path = "MODEL"

template = """You are a helpful AI assistant.

{history}
User: {question}
Assistant:"""

prompt = PromptTemplate.from_template(template)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# GPU
# Values based on the model and VRAM pool.
# n_gpu_layers = 40
# n_batch = 512
# 
# llm = LlamaCpp(
#     model_path=model_path,
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     callback_manager=callback_manager,
#     verbose=True,
#     temperature=0.75
# )

# Cpu
llm = LlamaCpp(
    model_path=model_path,
    temperature=0,
    max_tokens=200,
    top_p=1,
    callback_manager=callback_manager,
    verbose=False,  # Verbose is required to pass to the callback manager
)

llm_chain = prompt | llm

history = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    formatted_history = "\n".join(history)
    
    print("\nAssistant: ", flush=True)

    response = llm_chain.invoke({
        "history": formatted_history,
        "question": user_input
    })

    history.append(f"User: {user_input}")
    history.append(f"Assistant: {response.strip()}")

