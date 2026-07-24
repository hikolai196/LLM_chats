# LLM Chat & Completion Suite

This repository provides various chat and completion interfaces for Large Language Models (LLMs), focusing mainly on open-source models and adaptable platform integrations.

---
## Features
- LLM chat and completion in multiple forms
- Support for open-source models via Ollama, LLAMACpp, and more
- Platform integrations: Groq, Azure OpenAI, Google Gemini
- Embedding and streaming capabilities

---
## Instructions & Dependencies

**General Requirements:**
- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/) (for chatbot UI)

### 1. Ollama
- Required for:
  - Ollama_agent

### 2. LLAMACPP
- Required for:
  - LLAMACPP_LC.py

### 3. APIs
- google_gemini_chat; Requires Google AI API
- AzureAPI; Requires Azure OpenAI API
- GroqChat.py &rarr; Requires Groq platform API

---
## Installation

1. **Clone the Repository**
   
bash
```
git clone https://github.com/hikolai196/LLM_chats.git
cd llm-chat-suite
```

2. Install Python Dependencies

   `pip install ......`

3. Install Ollama 
  - Follow Ollama installation instructions. 
4. Set up Platform Credentials
  - For Groq, Azure OpenAI, and Google Gemini, ensure API keys and environment variables are configured.

--- 
## Usage
- Run chat/completion scripts with required dependencies and platforms
- Examples: 

   `python ......` or

   `streamlit run ......` or

  `uv run ......`

--- 
## Contributing

Contributions and suggestions are welcome!
Please fork the repo and submit a pull request or open an issue for discussion.

--- 
## License

This project is licensed under the MIT License. See LICENSE for details.

--- 
## Credits
- Developed by Yen-Ting "Hiko" Lai
- Built on open-source LLM and platform APIs
