# %%
import os
import glob
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# the above err due to usage of py 3.14 in env

# %%
load_dotenv()
GROQ_KEY = os.environ.get("GROQ_API_KEY")

model_name="meta-llama/llama-4-scout-17b-16e-instruct"

# %%
client = Groq() # GROQ_API_KEY

judge_sys_prompt = (
                    "You are an unbiased judge. "
                    "You will be given two answers: real_ans and model_ans. "
                    "Return ONLY a single integer percentage (0-100) "
                    "representing how semantically similar model_ans is to real_ans."
                )

def judge_llm(judge_prompt) -> float:  
  completion = client.chat.completions.create(
    model="moonshotai/kimi-k2-instruct-0905",
    messages=[
      {"role": "system", "content": f"{judge_sys_prompt}"},
      {"role": "user", "content": f"{judge_prompt}"}
    ],
    temperature=0,
    max_completion_tokens=3, # 0 to 100
    top_p=1,
    stream=False
  )

  return completion.choices[0].message

# %%
real = "Python supports asynchronous programming using asyncio."
model = "Python allows async programming via the asyncio library."

score = judge_llm(
    real_ans=real,
    model_ans=model
)

print(score.content)  # e.g., "92"


# %%
PDF_DIR = Path("docs")
OUT_DIR = Path("extracted_docs")

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

process_pdfs(PDF_DIR, OUT_DIR)

# %%
def make_documents(text: str, source: str):
    return [
        Document(
            page_content=text,
            metadata={"source": source}
        )
    ]


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vectorstore(chunks, embeddings):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name="docs_rag"
    )


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# %%
# this takes time!

all_docs = []

for text in glob.glob("extracted_docs/*.txt"):
    with open(text, 'r') as txt:
        docs = make_documents(txt.read(), text)
        all_docs.extend(docs)

chunks = chunk_documents(all_docs)
embeddings = get_embeddings()
vectorstore = build_vectorstore(chunks, embeddings)

# %%
# vectorstore.save_local("faiss_index")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="docs_rag"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# %%
llm = ChatGroq(
    model=model_name,
    temperature=0.0
)

# %%
prompt = ChatPromptTemplate.from_template("""
You are a factual assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")


# %%
# chaining
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

rag_question = "tell me about Bank of Tokyo Mitsubishi Ltd."

response = rag_chain.invoke(str(rag_question))
print()
print(rag_question)
print("RAG reply: ", response.content)

# %%
retriever.invoke(rag_question)

# %%
import pandas as pd

df = pd.read_csv('qna.csv')
df.head()

# %%
df['model_ans'] = [rag_chain.invoke(str(qn)).content for qn in df['qn']]

# %%
df.to_csv("ans.csv", index=False)

# %%
with_ans_df = pd.read_csv('ans.csv')
with_ans_df

# %%
with_ans_df['real_ans'][0]

# %%
ans_similarities, judge_scores = [], []

for real_ans, model_ans in zip(with_ans_df['real_ans'], with_ans_df['model_ans']):

    real_ans_embedding = embeddings.embed_query(real_ans)
    model_ans_embeddings = embeddings.embed_query(model_ans)
    sim = cosine_similarity([real_ans_embedding], [model_ans_embeddings])[0][0]
    ans_similarities.append(float(round(sim, 4) * 100))


    judge_prompt = f"""
    real_ans:
    {real_ans}

    model_ans:
    {model_ans}

    Return only a number from 0 to 100.
    """
    message = judge_llm(judge_prompt)
    judge_scores.append(message.content)

# %%
with_ans_df['ans_cos_sim'] = ans_similarities
with_ans_df['judge_scores'] = judge_scores

with_ans_df

# %%
with_ans_df.to_csv("ans_with_machine_scores.csv", index=False)

# %%



