import ollama
import pandas as pd
import chromadb
import numpy as np
import os

class OllamaEmbedding:
    def __init__(self, model="MODELHERE"):
        self.model = model

    def __call__(self, input):
        if isinstance(input, str):
            texts = [input]
        elif isinstance(input, list):
            texts = input
        else:
            raise ValueError("Input must be a string or a list of strings")

        embeddings = []
        for text in texts:
            response = ollama.embeddings(model=self.model, prompt=text)
            embeddings.append(response['embedding'])
        
        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings

def load_csv_to_chroma(csv_path, collection_name):
    # Load CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize ChromaDB client
    chroma_client = chromadb.Client()
    
    # Create a new collection (or get existing one)
    embedding_function = OllamaEmbedding()
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    # Prepare documents, ids, and metadata
    documents = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()
    ids = [str(i) for i in range(len(documents))]
    metadatas = df.to_dict('records')
    
    # Add documents to the collection
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    
    return collection

def query_chroma(collection, query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results['documents'][0]

def chatbot(csv_path):
    # Load CSV into ChromaDB
    collection_name = os.path.basename(csv_path).split('.')[0]
    collection = load_csv_to_chroma(csv_path, collection_name)
    
    history = []
    print("Chatbot: Hello! I'm a chatbot powered by Llama 3 with RAG. How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        # Query ChromaDB for relevant information
        relevant_info = query_chroma(collection, user_input)
        
        # Prepare the prompt with history and relevant information
        prompt = "You are a helpful AI assistant with access to a database of information. "
        prompt += "Use the following relevant information to inform your response, but don't refer to the information directly unless necessary:\n\n"
        prompt += "Relevant Information:\n" + "\n".join(relevant_info) + "\n\n"
        prompt += "Conversation History:\n"
        for entry in history:
            prompt += f"{entry['role']}: {entry['content']}\n"
        prompt += f"Human: {user_input}\nAI:"
        
        # Generate response
        response = ollama.generate(model='gemma2:2b', prompt=prompt)
        
        print(f"Chatbot: {response['response']}")
        
        # Update history
        history.append({"role": "Human", "content": user_input})
        history.append({"role": "AI", "content": response['response']})
        
        # Limit history to last 5 exchanges to manage context length
        if len(history) > 10:
            history = history[-10:]

if __name__ == "__main__":
    csv_path = "FILEHERE"
    chatbot(csv_path)