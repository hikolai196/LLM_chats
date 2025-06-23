import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = "MODEL"

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Configure the Gemini API with your API key
genai.configure(api_key=api_key)

def initialize_chat():
    """Initialize a new chat session with Gemini."""
    # Create a GenerativeModel with the Gemini Pro model
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
        }
    )
    
    # Start a new chat session
    chat = model.start_chat(history=[])
    
    return chat

def run_chatbot():
    chat = initialize_chat()
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGemini: Goodbye! Have a great day!")
            break
        
        try:
            # Send user message to Gemini and get response
            response = chat.send_message(user_input)
            
            # Print the response
            print(f"\nGemini: {response.text}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            
if __name__ == "__main__":
    run_chatbot()