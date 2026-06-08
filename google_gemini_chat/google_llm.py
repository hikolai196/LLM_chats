import os

import google.generativeai as genai
from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
}


def get_settings():
    """Load Gemini settings from environment variables."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("MODEL", DEFAULT_MODEL)

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    return api_key, model_name


def build_system_instruction(document_context=None):
    """Create instructions for document-aware chat sessions."""
    if not document_context:
        return None

    return (
        "You are a helpful assistant. Use the uploaded document context "
        "to answer questions and analyze the documents. If the answer is "
        "not in the provided documents, say that clearly.\n\n"
        f"Uploaded document context:\n{document_context}"
    )


def initialize_chat(generation_config=None, document_context=None):
    """Initialize a new chat session with Gemini."""
    api_key, model_name = get_settings()
    genai.configure(api_key=api_key)

    config = DEFAULT_GENERATION_CONFIG.copy()
    if generation_config:
        config.update(generation_config)

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=config,
        system_instruction=build_system_instruction(document_context),
    )

    return model.start_chat(history=[])


def send_message(chat, message):
    """Send a message to Gemini and return the response text."""
    response = chat.send_message(message)
    return response.text

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
            response_text = send_message(chat, user_input)
            print(f"\nGemini: {response_text}")

        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    run_chatbot()