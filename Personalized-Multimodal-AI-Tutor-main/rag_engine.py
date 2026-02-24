import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import GPT4All
import torch

def process_pdf(pdf_file):
    """
    Extracts text from a uploaded PDF file.
    """
    if pdf_file is None:
        return ""
    
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def create_vector_db(text):
    """
    Creates a FAISS vector database from text.
    Using 'sentence-transformers/all-MiniLM-L6-v2' for embeddings (CPU friendly).
    """
    if not text:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Use a small, efficient model for embeddings on CPU
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def get_llm_chain(vector_store, model_path=None):
    """
    Creates a RetrievalQA chain.
    If model_path is provided, uses local GPT4All model.
    Else, defaults to GPT-2 from HuggingFace.
    """
    llm = None
    
    if model_path and os.path.exists(model_path):
        print(f"Loading local model from: {model_path}")
        # GPT4All loader
        llm = GPT4All(
            model=model_path,
            backend="gptj", # or 'llama', auto-detection usually works or is default
            verbose=True
        )
    else:
        # Fallback / Default: GPT-2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using fallback device: {device}")
        model_id = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            model_kwargs={"temperature": 0.7},
            device=0 if device == "cuda" else -1
        )
        llm = HuggingFacePipeline(pipeline=pipe)

    # Create a custom prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain
