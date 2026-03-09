from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import time
import sys
import re

CHROMA_PATH = "./chroma_db"
LLM_MODEL = "qwen2.5:1.5b"
TOP_K       = 3
CANDIDATE_K = 20

PROMPT_TEMPLATE = """
You are a highly efficient technical assistant. 

CRITICAL INSTRUCTIONS:
1. Answer the question directly and accurately using ONLY the provided context.
2. DO NOT use introductory filler phrases (e.g., "The context states", "According to the documents", or "Yes, that is correct"). Start your answer immediately.
3. Keep your response strictly under 3 sentences. Be as concise as possible.
4. If the context does not contain the exact answer, reply ONLY with: "I don't know."

Context:
{context}

Question: {question}

Answer:"""


def clean_text(text):
    return re.sub(r'[.,]', ' ', text.lower()).split()


class VectorFirstHybridRetriever:
    """Vector search gets candidates, BM25 reranks only those — not all 25k chunks."""
    
    def __init__(self, vector_retriever, weights=(0.3, 0.7), k=TOP_K, candidate_k=CANDIDATE_K):
        self.vector_retriever = vector_retriever
        self.bm25_weight = weights[0]
        self.vector_weight = weights[1]
        self.k = k
        self.candidate_k = candidate_k
    
    def invoke(self, query):
        candidates = self.vector_retriever.invoke(query)
        
        tokenized_corpus = [clean_text(doc.page_content) for doc in candidates]
        tokenized_query = clean_text(query)
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        doc_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(candidates):
            key = doc.page_content[:100]
            doc_map[key] = doc
            doc_scores[key] = self.vector_weight / (rank + 60)
        
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        for i, doc in enumerate(candidates):
            key = doc.page_content[:100]
            normalized_bm25 = bm25_scores[i] / max_bm25
            doc_scores[key] += self.bm25_weight * normalized_bm25 / 60
        
        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys[:self.k]]


def load_retriever():
    print("Loading Vector Store (Semantic Search)...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=2)
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "ip"}
    )
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CANDIDATE_K}
    )

    print("Building Vector-First Hybrid Retriever...")
    return VectorFirstHybridRetriever(
        vector_retriever=vector_retriever,
        weights=(0.3, 0.7),
        k=TOP_K,
        candidate_k=CANDIDATE_K
    )


def ask(query, retriever, llm):
    retrieval_start = time.time()
    docs = retriever.invoke(query)
    retrieval_time = (time.time() - retrieval_start) * 1000

    if not docs:
        print("\n🤖 Answer: I don't know (no relevant documents found).")
        print(f"⏱️ Retrieval: {retrieval_time:.0f}ms")
        return

    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context, question=query
    )

    print(f"\n⏱️ Retrieval: {retrieval_time:.0f}ms | Docs: {len(docs)}")

    print("\n📎 Top 3 Retrieved Chunks:")
    for i, doc in enumerate(docs[:3]):
        source = doc.metadata.get('source', 'Unknown')
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"  [{i+1}] ({source})")
        print(f"      {snippet}...")

    print(f"\n🤖 Answer: ", end="", flush=True)

    llm_start = time.time()
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)
    llm_time = (time.time() - llm_start) * 1000

    print(f"\n⏱️ LLM: {llm_time:.0f}ms | Total: {retrieval_time + llm_time:.0f}ms")

    print("\n\n📄 Sources:")
    seen = set()
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        if source not in seen:
            seen.add(source)
            print(f"  - {source}")


if __name__ == "__main__":
    print("=" * 40)
    print("   RAG QUERY PIPELINE (STREAMING)")
    print("=" * 40)
    print("\nLoading models, please wait...\n")

    retriever = load_retriever()
    
    # Warm up the embedding model — first call loads ONNX into memory
    print("Warming up embedding model...")
    _ = retriever.invoke("warmup query")
    
    llm = OllamaLLM(
        model=LLM_MODEL, 
        temperature=0.2,
        num_predict=256,
        num_ctx=1024  
    )

    print("\n✅ System Ready! Type 'quit' to exit.\n")

    while True:
        query = input("❓ Ask: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            sys.exit()
        if query:
            ask(query, retriever, llm)
            print()