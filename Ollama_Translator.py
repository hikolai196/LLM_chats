# Chat completion
import tkinter as tk
from tkinter import ttk
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

model="MODEL"

llm = ChatOllama(
    model=model,
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a translator, only translates {input} frpm {input_language} into {output_language} and only output {output_language}.",
    ),
    ("human", "{input}"),
])

chain = prompt | llm

response = chain.invoke({
    "input_language": "English",
    "output_language": "Japanese",
    "input": "What do you mean.",
})

print(response.content)

# with tkinter
def translate():
    input_language = input_language_var.get()
    output_language = output_language_var.get()
    input_text = input_text_var.get()
    
    response = chain.invoke({
        "input_language": input_language,
        "output_language": output_language,
        "input": input_text,
    })
    
    output_text_var.set(response.content)

# Create the main window
root = tk.Tk()
root.title("Language Translator")

# Create and place widgets
tk.Label(root, text="Input Language:").grid(row=0, column=0, padx=10, pady=10)
input_language_var = tk.StringVar(value="English")
input_language_menu = ttk.Combobox(root, textvariable=input_language_var)
input_language_menu['values'] = ("English", "Japanese", "Spanish", "Chinese")
input_language_menu.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Output Language:").grid(row=1, column=0, padx=10, pady=10)
output_language_var = tk.StringVar(value="Japanese")
output_language_menu = ttk.Combobox(root, textvariable=output_language_var)
output_language_menu['values'] = ("English", "Japanese", "Spanish", "Chinese")
output_language_menu.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="Input Text:").grid(row=2, column=0, padx=10, pady=10)
input_text_var = tk.StringVar()
input_text_entry = tk.Entry(root, textvariable=input_text_var, width=50)
input_text_entry.grid(row=2, column=1, padx=10, pady=10)

tk.Label(root, text="Output Text:").grid(row=3, column=0, padx=10, pady=10)
output_text_var = tk.StringVar()
output_text_entry = tk.Entry(root, textvariable=output_text_var, width=50, state='readonly')
output_text_entry.grid(row=3, column=1, padx=10, pady=10)

translate_button = tk.Button(root, text="Translate", command=translate)
translate_button.grid(row=4, column=0, columnspan=2, pady=10)

# Run the main loop
root.mainloop()