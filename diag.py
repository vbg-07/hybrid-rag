from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
import time
import re

CHROMA_PATH = "./chroma_db"
TOP_K = 3


class ParallelHybridRetriever:
    """Drop-in replacement for EnsembleRetriever with parallel execution."""
    
    def __init__(self, bm25_retriever, vector_retriever, weights=(0.3, 0.7), k=TOP_K):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.bm25_weight = weights[0]
        self.vector_weight = weights[1]
        self.k = k
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def invoke(self, query):
        future_bm25 = self._executor.submit(self.bm25_retriever.invoke, query)
        future_vector = self._executor.submit(self.vector_retriever.invoke, query)
        
        bm25_docs = future_bm25.result()
        vector_docs = future_vector.result()
        
        doc_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + self.bm25_weight / (rank + 60)
        
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content[:100]
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + self.vector_weight / (rank + 60)
        
        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys[:self.k]]


def load_retriever():
    print("Loading Vector Store (Semantic Search)...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=2)
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "ip"}  # inner product for normalized BGE embeddings
    )
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    print("Building BM25 Index (Exact Keyword Search)...")
    db_data = vectorstore.get()
    docs = []
    
    if db_data and 'documents' in db_data and db_data['documents']:
        for i in range(len(db_data['documents'])):
            meta = db_data['metadatas'][i] if db_data['metadatas'] else {}
            docs.append(Document(
                page_content=db_data['documents'][i], 
                metadata=meta
            ))
            
    if not docs:
        print("Database is empty.")
        return vector_retriever
    
    def clean_text(text):
        return re.sub(r'[.,]', ' ', text.lower()).split()

    bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=clean_text)
    bm25_retriever.k = TOP_K

    print("Combining into Parallel Hybrid Retriever...")
    return ParallelHybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=(0.3, 0.7),
        k=TOP_K
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