import streamlit as st
import os
import time
from rag_engine import process_pdf, create_vector_db, get_llm_chain
from image_gen import get_image_generator, generate_diagram

st.set_page_config(page_title="AI Tutor", layout="wide")

st.title("🤖 Personalized Multimodal AI Tutor")
st.markdown("Upload your lecture notes (PDF) and ask questions. I can also draw diagrams!")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "image_pipe" not in st.session_state:
    st.session_state.image_pipe = None
if "db_processed" not in st.session_state:
    st.session_state.db_processed = False

# Sidebar for Setup
with st.sidebar:
    st.header("📚 Study Material")
    model_path = st.text_input("Local Model Path (Optional)", value="", help="Path to your .gguf model file")
    uploaded_file = st.file_uploader("Upload PDF Lecture Notes", type=["pdf"])
    
    if uploaded_file and not st.session_state.db_processed:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF and building Knowledge Base... (This may take a moment on CPU)"):
                # extracting text
                text = process_pdf(uploaded_file)
                st.write(f"extracted {len(text)} characters.")
                
                # create db
                vector_store = create_vector_db(text)
                if vector_store:
                    # setup chain
                    st.session_state.qa_chain = get_llm_chain(vector_store, model_path=model_path)
                    st.session_state.db_processed = True
                    st.success("PDF Processed! You can now ask questions.")
                else:
                    st.error("Could not extract text or create DB.")

    st.header("🎨 Image Gen Setup")
    if st.session_state.image_pipe is None:
        if st.button("Load Image Model (Heavy)"):
            with st.spinner("Loading Stable Diffusion..."):
                st.session_state.image_pipe = get_image_generator()
            st.success("Image Model Loaded!")
    else:
        st.info("Image Model Active")

# Chat Interface - Legacy Compatibility
for message in st.session_state.messages:
    with st.container():
        st.markdown(f"**{message['role'].title()}:**")
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], caption="Generated Diagram")
        st.divider()

with st.form(key="chat_form", clear_on_submit=True):
    prompt = st.text_input("Ask a question...", key="input_field")
    submit_button = st.form_submit_button("Send")

if submit_button and prompt:
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Force rerun to show user message immediately
    st.experimental_rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    
    # Check if user wants an image
    generate_image = "draw" in prompt.lower() or "diagram" in prompt.lower() or "image" in prompt.lower()

    # AI Response
    with st.spinner("Thinking..."):
        response_text = ""
        generated_img = None
        
        # 1. Text Answer (RAG)
        if st.session_state.qa_chain:
            try:
                # Some versions might return just string, others dict
                res = st.session_state.qa_chain({"query": prompt})
                response_text = res["result"]
            except Exception as e:
                response_text = f"Error generating answer: {e}"
        else:
            response_text = "Please upload and process a PDF first to ask questions about it."

        # 2. Image Generation (if requested)
        if generate_image:
            if st.session_state.image_pipe:
                with st.spinner("Drawing diagram... (This is slow on CPU)"):
                    try:
                        generated_img = generate_diagram(st.session_state.image_pipe, prompt)
                    except Exception as e:
                        response_text += f"\n\n[Image Generation Failed: {e}]"
            else:
                response_text += "\n\n(Image model not loaded. Check sidebar to enable.)"

        # Save to history
        msg_data = {"role": "assistant", "content": response_text}
        if generated_img:
            msg_data["image"] = generated_img
        st.session_state.messages.append(msg_data)
        st.experimental_rerun()
