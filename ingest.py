import json
import os
import shutil
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_PATH = "./scifact/corpus.jsonl"
CHROMA_PATH = "./chroma_db"


def ingest():
    if os.path.exists(CHROMA_PATH):
        print("Clearing old database...")
        shutil.rmtree(CHROMA_PATH)

    print("Loading BEIR SciFact corpus...")
    documents = []
    with open(CORPUS_PATH) as f:
        for line in f:
            data = json.loads(line)
            doc_id = data["_id"]
            title = data.get("title", "")
            text = data.get("text", "")
            content = f"{title}\n{text}".strip()
            documents.append(Document(
                page_content=content,
                metadata={"doc_id": doc_id, "title": title}
            ))

    print(f"Loaded {len(documents)} documents.")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings and storing in ChromaDB...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=2)

    BATCH_SIZE = 5000
    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                batch,
                embeddings,
                persist_directory=CHROMA_PATH,
                collection_metadata={"hnsw:space": "ip"}
            )
        else:
            vectorstore.add_documents(batch)
        print(f"  Ingested {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks...")

    print(f"\nDatabase created at '{CHROMA_PATH}' with {len(chunks)} entries.")


if __name__ == "__main__":
    ingest()