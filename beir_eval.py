import json
import time
import re
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

CHROMA_PATH = "./chroma_db"
QRELS_PATH = "./scifact/qrels/test.tsv"
QUERIES_PATH = "./scifact/queries.jsonl"

TOP_K = 5  # Reduced from 7 — still captures relevant docs, faster retrieval


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


def setup_retriever():
    print("Loading Vector Store and building BM25 Index...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=2)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "ip"}  # inner product for normalized BGE embeddings
    )
    
    db_data = vectorstore.get()
    docs = [Document(page_content=txt, metadata=meta) for txt, meta in zip(db_data['documents'], db_data['metadatas'])]

    def clean_text(text):
        return re.sub(r'[.,]', ' ', text.lower()).split()

    bm25_retriever = BM25Retriever.from_documents(docs, preprocess_func=clean_text)
    bm25_retriever.k = TOP_K
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    return ParallelHybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=(0.3, 0.7),
        k=TOP_K
    )


def load_qrels():
    qrels = {}
    with open(QRELS_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def load_queries():
    queries = {}
    with open(QUERIES_PATH) as f:
        for line in f:
            data = json.loads(line)
            queries[data["_id"]] = data["text"]
    return queries


def evaluate():
    retriever = setup_retriever()
    
    # Warm up with multiple queries to fully heat HNSW cache
    print("Warming up embedding model...")
    warmup_queries = [
        "warmup query one",
        "warmup query two", 
        "warmup query three"
    ]
    for q in warmup_queries:
        _ = retriever.invoke(q)
    
    qrels = load_qrels()
    queries = load_queries()

    hits = 0
    total = 0
    total_time = 0

    eval_queries = {qid: queries[qid] for qid in qrels if qid in queries}
    print(f"\nEvaluating {len(eval_queries)} queries...\n")

    for qid, query_text in eval_queries.items():
        start = time.time()
        results = retriever.invoke(query_text)
        elapsed = time.time() - start
        total_time += elapsed

        retrieved_ids = set()
        for doc in results:
            doc_id = doc.metadata.get("doc_id", "")
            if doc_id:
                retrieved_ids.add(str(doc_id))

        relevant = qrels[qid]
        if retrieved_ids & relevant:
            hits += 1
        total += 1

    accuracy = hits / total * 100 if total else 0
    avg_time = (total_time / total * 1000) if total else 0

    print("=" * 40)
    print(f"  Queries Evaluated : {total}")
    print(f"  Hits (≥1 relevant): {hits}")
    print(f"  Accuracy (Hit Rate): {accuracy:.2f}%")
    print(f"  Avg Retrieval Time : {avg_time:.0f}ms")
    print("=" * 40)


if __name__ == "__main__":
    evaluate()