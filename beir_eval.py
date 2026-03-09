import json
import time
import re
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

CHROMA_PATH = "./chroma_db"
QRELS_PATH = "./scifact/qrels/test.tsv"
QUERIES_PATH = "./scifact/queries.jsonl"

TOP_K = 5
CANDIDATE_K = 30  # Vector retrieves this many, BM25 reranks them


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
        # Step 1: Vector search gets top candidates (fast HNSW)
        candidates = self.vector_retriever.invoke(query)
        
        # Step 2: BM25 rerank ONLY these candidates (not 25k chunks)
        tokenized_corpus = [clean_text(doc.page_content) for doc in candidates]
        tokenized_query = clean_text(query)
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Step 3: Weighted fusion of vector rank + BM25 score
        doc_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(candidates):
            key = doc.page_content[:100]
            doc_map[key] = doc
            # Vector rank score
            doc_scores[key] = self.vector_weight / (rank + 60)
        
        # Normalize BM25 scores and add
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        for i, doc in enumerate(candidates):
            key = doc.page_content[:100]
            normalized_bm25 = bm25_scores[i] / max_bm25
            doc_scores[key] += self.bm25_weight * normalized_bm25 / 60
        
        sorted_keys = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [doc_map[k] for k in sorted_keys[:self.k]]


def setup_retriever():
    print("Loading Vector Store...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=2)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "ip"}
    )
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": CANDIDATE_K})

    return VectorFirstHybridRetriever(
        vector_retriever=vector_retriever,
        weights=(0.3, 0.7),
        k=TOP_K,
        candidate_k=CANDIDATE_K
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