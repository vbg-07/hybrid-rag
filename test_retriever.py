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
    print("🔄 Loading Vector Store (Semantic Search)...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=2)

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    print("🔄 Building BM25 Index (Exact Keyword Search)...")
    db_data = vectorstore.get()
    docs = []
    
    if db_data and 'documents' in db_data:
        for i in range(len(db_data['documents'])):
            docs.append(Document(
                page_content=db_data['documents'][i], 
                metadata=db_data['metadatas'][i]
            ))
            
    if not docs:
        print("⚠️ Database is empty. Returning basic vector retriever.")
        return vector_retriever

    def clean_text(text):
        return re.sub(r'[.,]', ' ', text.lower()).split()

    bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=clean_text)
    bm25_retriever.k = TOP_K

    retriever = ParallelHybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=(0.3, 0.7),
        k=TOP_K
    )

    # Warm up
    print("Warming up...")
    _ = retriever.invoke("warmup query")

    test_queries = [
        "Calf serum proteins are altered by heat treatment.",
        "Mutations in the__(gene) gene cause X-linked adrenoleukodystrophy.",
        "0-3 cM is the average distance between markers in a genetic map.",
    ]

    print(f"\nTesting with {len(test_queries)} queries...\n")

    for query in test_queries:
        start = time.time()
        results = retriever.invoke(query)
        elapsed = (time.time() - start) * 1000

        print(f"Query: {query}")
        print(f"Time: {elapsed:.0f}ms | Results: {len(results)}")
        for doc in results:
            print(f"  - {doc.page_content[:80]}...")
        print()

    return retriever

if __name__ == "__main__":
    print("=" * 50)
    print("   🔍 RAG RETRIEVAL TESTER (NO LLM)")
    print("=" * 50 + "\n")
    
    retriever = load_retriever()
    print("\n✅ Retriever Ready! Type your query to see what it finds.")
    print("   Type 'quit' to exit.\n")
    
    while True:
        query = input("❓ Search Query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
            
        start_time = time.time()
        
        # We invoke the retriever directly, skipping the LLM entirely
        results = retriever.invoke(query)
        
        # De-duplicate just like we do in query.py
        seen = set()
        unique_results = []
        for chunk in results:
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                unique_results.append(chunk)
        results = unique_results
        
        search_time = time.time() - start_time
        
        print(f"\n⏱️  Search completed in {search_time*1_000_000:.2f}µs")
        print(f"📄 Top {len(results)} chunks retrieved:")
        print("-" * 50)
        
        for i, doc in enumerate(results):
            print(f"\n--- CHUNK {i+1} ---")
            
            # Format the metadata to show off your new Markdown headers!
            meta = doc.metadata
            source = os.path.basename(meta.get("source", "Unknown Source"))
            print(f"📁 Source: {source}")
            
            # Extract and print any Markdown headers if the splitter caught them
            headers = [f"{k}: {v}" for k, v in meta.items() if k.startswith("Header")]
            if headers:
                print(f"🏷️  Headers: {' | '.join(headers)}")
                
            # Print the content preview cleanly
            preview = doc.page_content.replace('\n', ' ')
            if len(preview) > 300:
                preview = preview[:300] + "..."
            print(f"📝 Content:\n{preview}\n")
        print("-" * 50)