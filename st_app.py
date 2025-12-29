import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf_text(file):
    reader = PdfReader(file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def build_faiss(chunks, model):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve(query, chunks, index, model, k=3):
    q_emb = model.encode([query], convert_to_numpy=True)
    _, idx = index.search(q_emb, k)
    return [chunks[i] for i in idx[0]]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="PDF Chat", layout="centered")
st.title("📄 Chat with your PDF")

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and "index" not in st.session_state:
    with st.spinner("Processing PDF..."):
        text = load_pdf_text(uploaded_file)
        chunks = chunk_text(text)
        index, _ = build_faiss(chunks, model)

        st.session_state.chunks = chunks
        st.session_state.index = index

        st.success("PDF loaded!")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "index" not in st.session_state:
            answer = "Please upload a PDF first."
        else:
            docs = retrieve(
                prompt,
                st.session_state.chunks,
                st.session_state.index,
                model,
            )
            # Very naive "answer"
            answer = "### Relevant excerpts:\n\n" + "\n\n---\n\n".join(docs)

        st.markdown(answer)
        

    st.session_state.messages.append({"role": "assistant", "content": answer})
