from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import time
import re

CHROMA_PATH = "./chroma_db"
TOP_K = 3
CANDIDATE_K = 20


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


if __name__ == "__main__":
    print("=" * 40)
    print("   DIAGNOSTIC: RETRIEVER TEST")
    print("=" * 40)
    
    retriever = load_retriever()
    
    # Warm up
    print("\nWarming up embedding model...")
    _ = retriever.invoke("warmup query")
    
    test_queries = [
        "Calf serum proteins are altered by heat treatment.",
        "Mutations in the__(gene) gene cause X-linked adrenoleukodystrophy.",
        "0-3 cM is the average distance between markers in a genetic map.",
    ]
    
    print(f"\nRunning {len(test_queries)} test queries...\n")
    
    for query in test_queries:
        start = time.time()
        results = retriever.invoke(query)
        elapsed = (time.time() - start) * 1000
        
        print(f"Q: {query}")
        print(f"   ⏱️ {elapsed:.0f}ms | {len(results)} results")
        for doc in results:
            source = doc.metadata.get('source', 'Unknown')
            print(f"   📄 [{source}] {doc.page_content[:80]}...")
        print()