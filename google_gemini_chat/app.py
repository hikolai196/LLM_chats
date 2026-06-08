import streamlit as st

from document_loader import build_document_context, load_document
from google_llm import DEFAULT_GENERATION_CONFIG, initialize_chat, send_message


st.set_page_config(page_title="Gemini Chatbot", page_icon=":speech_balloon:")
st.title("Gemini Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = None

if "chat_signature" not in st.session_state:
    st.session_state.chat_signature = None


def get_chat(generation_config, document_context):
    if st.session_state.chat is None:
        st.session_state.chat = initialize_chat(
            generation_config=generation_config,
            document_context=document_context,
        )
    return st.session_state.chat


with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload files for analysis",
        type=["csv", "docx", "txt", "xls", "xlsx"],
        accept_multiple_files=True,
    ) or []

    st.header("LLM Parameters")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(DEFAULT_GENERATION_CONFIG["temperature"]),
        step=0.05,
    )
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULT_GENERATION_CONFIG["top_p"]),
        step=0.01,
    )
    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=100,
        value=int(DEFAULT_GENERATION_CONFIG["top_k"]),
        step=1,
    )
    max_output_tokens = st.slider(
        "Max output tokens",
        min_value=256,
        max_value=8192,
        value=2048,
        step=256,
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.chat = None

generation_config = {
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "max_output_tokens": max_output_tokens,
}

loaded_documents = []
load_errors = []
for uploaded_file in uploaded_files:
    try:
        content = load_document(uploaded_file.name, uploaded_file.getvalue())
        if content.strip():
            loaded_documents.append(
                {"name": uploaded_file.name, "content": content}
            )
        else:
            load_errors.append(f"{uploaded_file.name}: no readable text found")
    except Exception as exc:
        load_errors.append(f"{uploaded_file.name}: {exc}")

document_context = (
    build_document_context(loaded_documents)
    if loaded_documents
    else None
)

with st.sidebar:
    if loaded_documents:
        st.success(f"Loaded {len(loaded_documents)} document(s).")
        for document in loaded_documents:
            st.caption(f"{document['name']} ({len(document['content']):,} chars)")

    for load_error in load_errors:
        st.warning(load_error)

chat_signature = (
    tuple((document["name"], len(document["content"])) for document in loaded_documents),
    tuple(sorted(generation_config.items())),
)
if st.session_state.chat_signature != chat_signature:
    st.session_state.chat = None
    st.session_state.chat_signature = chat_signature

if document_context:
    st.info(
        "Uploaded documents are loaded as context. Ask the assistant to "
        "summarize, compare, extract, or analyze them."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask Gemini something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response_text = send_message(
                    get_chat(generation_config, document_context),
                    prompt,
                )
            st.markdown(response_text)
        except Exception as exc:
            response_text = f"Error: {exc}"
            st.error(response_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
