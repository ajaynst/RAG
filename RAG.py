import os
import glob
from dotenv import load_dotenv
from pathlib import Path
import fitz  # PyMuPDF
from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
GROQ_KEY = os.environ.get("GROQ_API_KEY")

model_name="meta-llama/llama-4-scout-17b-16e-instruct"

PDF_DIR = Path("books")
OUT_DIR = Path("extracted_text")

client = Groq() # GROQ_API_KEY

def call_groq(model_name, user_message):  
  completion = client.chat.completions.create(
    model=model_name,
    messages=[
      {
        "role": "user",
        "content": f"{user_message}"
      }
    ],
    temperature=0,
    max_completion_tokens=8192,
    top_p=1,
    # reasoning_effort="medium", # not supported for llama 3.3 70B
    stream=False,
    stop=None
  )

  return completion.choices[0].message

OUT_DIR.mkdir(exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"\n--- Page {i + 1} ---\n{text}")

    return "\n".join(pages)


def process_pdfs(pdf_dir: Path, out_dir: Path):
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Extracting: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)

        output_file = out_dir / f"{pdf_file.stem}.txt"
        output_file.write_text(text, encoding="utf-8")

def make_documents(text: str, source: str):
    return [
        Document(
            page_content=text,
            metadata={"source": source}
        )
    ]


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# this takes time!
all_docs = []

for text in glob.glob("extracted_text/*.txt"):
    with open(text, 'r') as txt:
        docs = make_documents(txt.read(), text)
        all_docs.extend(docs)

# process_pdfs(PDF_DIR, OUT_DIR)

chunks = chunk_documents(all_docs)
embeddings = get_embeddings()
vectorstore = build_vectorstore(chunks, embeddings)

vectorstore.save_local("faiss_index")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

llm = ChatGroq(
    model=model_name,
    temperature=0.0
)

prompt = ChatPromptTemplate.from_template("""
You are a factual assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

# chaining
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# "explain start with why book in 10 sentences."
# "tell me about a business book"
rag_question = "who is simon sinek?"

response = rag_chain.invoke(rag_question)
print(response)
print()
print(rag_question)
print("RAG reply: ", response.content)

message = call_groq(model_name=model_name, user_message=rag_question)

print("Model Response: ", message.content)

# Ultimate Test
ultimate_questions = ["Who is Ajay?", "Which shell does ajay use, and why?", "which is the programming lang used by ajay?"]

for ultimate_question in ultimate_questions:
    response = rag_chain.invoke(ultimate_question)

    print()

    print(ultimate_question)
    print("Model WITH RAG: ", response.content)

    print()

    ans = call_groq(model_name=model_name, user_message=ultimate_question)
    print("Model WITHOUT RAG: ", ans.content)
    print('-' * 40)