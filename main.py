import os
import re
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader  # <--- Add this import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


# Always use absolute path for pdfs directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdfs_directory = os.path.join(BASE_DIR, "pdfs")


# Read Ollama base URL from environment variable, default to localhost:11434
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)
model = OllamaLLM(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)


template = """
You are an expert assistant answering questions using only the provided company guideline excerpts.


INSTRUCTIONS:
- Your answer must begin with a single, concise markdown heading (## ...). Never use more than one heading.
- The rest of your answer must be formatted as clear bullet points (- ...) OR numbered lists (1., 2., ...), not as paragraphs or narrative text.
- Bold any key sections or article names using markdown (**...**).
- Do not include any <think> or internal thoughts.
- Do not repeat the heading.
- Be as thorough as needed—do not artificially limit the answer length.
- If the answer is not present in the material, state: "This information is not available in the provided company guidelines."
- Produce valid markdown only, with one heading and then a bullet/numbered list.


Question: {question}
Context: {context}


YOUR OUTPUT:
"""


def upload_pdf(file):
    """Saves an uploaded file to the pdfs directory. Returns absolute saved path."""
    os.makedirs(pdfs_directory, exist_ok=True)
    file_path = os.path.join(pdfs_directory, file.name)
    file.seek(0)
    with open(file_path, "wb") as f:
        f.write(file.read())
    print("Saved file:", file_path, "- Exists:", os.path.isfile(file_path))
    return file_path


def create_vector_store(file_path):
    """Loads PDF, chunks and indexes it, and returns the vector store."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(chunked_docs, embeddings)
    return db


def retrieve_docs(db, query, k=4):
    """
    Returns top-k relevant doc chunks, including ONLY page and section info, NOT the file path.
    """
    docs = db.similarity_search(query, k=k)
    enriched = []
    for doc in docs:
        page = ""
        section = ""
        if hasattr(doc, "metadata"):
            if isinstance(doc.metadata, dict):
                page = doc.metadata.get("page", "")
                # 'section' field if present in PDF metadata
                section = doc.metadata.get("section", "")
        enriched.append({
            "content": doc.page_content,
            "page": page,
            "section": section
        })
    return enriched


def question_pdf(question, documents):
    """
    Builds answer using retrieved docs and prompt; includes page/section only in sources.
    """
    context = "\n\n".join([d["content"] for d in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    answer = chain.invoke({"question": question, "context": context})

    # Remove any accidental <think> blocks if present.
    answer_clean = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    # Only include page and section in sources
    sources = []
    for d in documents:
        ref = {}
        if d["section"]:
            ref["section"] = d["section"]
        if d["page"]:
            ref["page"] = d["page"]
        ref["content"] = d["content"][:200] + ("..." if len(d["content"]) > 200 else "")
        sources.append(ref)
    return {"answer": answer_clean, "sources": sources}
