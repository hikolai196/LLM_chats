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

### 1. LangChain & Ollama
- Required for:
  - CodeAssistant.py
  - LLAMACpp_LC.py
  - OllamaChat.py
  - OllamaChat2.py (includes Embedding)
  - OllamaLLM.py

### 2. Ollama Only
- Required for:
  - OllamaSimple.py

### 3. Streamlit & Ollama
- Required for:
  - StreamlitChatbot.py

### 4. Platform-Specific Scripts
- GroqChat.py &rarr; Requires Groq platform
- AzureOpenAI.py &rarr; Requires Azure OpenAI
- google_gemini.py &rarr; Requires Google Gemini

---
## Installation

1. **Clone the Repository**
   
bash
```
git clone https://github.com/hikolai196/LLM_chats.git
cd llm-chat-suite
```

2. Install Python Dependencies

   `pip install langchain streamlit`

3. Install Ollama 
  - Follow Ollama installation instructions. 
4. Set up Platform Credentials
  - For Groq, Azure OpenAI, and Google Gemini, ensure API keys and environment variables are configured.

--- 
## Usage
- Run chat/completion scripts with required dependencies and platforms
- Example (Ollama + LangChain): 

   `python OllamaChat.py`

- Example (Streamlit chatbot): 

   `streamlit run StreamlitChatbot.py`

--- 
## Script Overview
- CodeAssistant.py: LLM-powered code assistant (LangChain, Ollama) 
- LLAMACpp_LC.py: LLAMACpp integration via LangChain 
- OllamaChat.py: Standard Ollama chat interface (LangChain) 
- OllamaChat2.py: Ollama chat with embedding support (LangChain) 
- OllamaLLM.py: LLM completion via Ollama (LangChain) 
- OllamaSimple.py: Minimal Ollama chat (Ollama only) 
- StreamlitChatbot.py: Web-based chatbot UI (Streamlit, Ollama) 
- GroqChat.py: Groq platform chat 
- AzureOpenAI.py: Azure OpenAI chat/completion 
- google_gemini.py: Google Gemini chat/completion

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
