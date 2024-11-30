from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# The model path
model_path = r"MODELHERE" 

template = """Question: {question}

Answer: """

prompt = PromptTemplate.from_template(template=template)

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
    verbose=True,  # Verbose is required to pass to the callback manager
)

llm_chain = prompt | llm

question = "請簡短推薦兩間台中美食"

print(llm_chain.invoke(question))
