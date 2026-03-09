# Hybrid RAG — High-Performance Local Retrieval-Augmented Generation

A fully local RAG pipeline optimized for scientific document retrieval, benchmarked on the **SciFact BEIR** dataset (300 queries). Achieves **77% accuracy** with **35ms avg retrieval** on consumer hardware — no GPU or cloud APIs required.

## Performance

| Metric | Value |
|---|---|
| Accuracy (Hit Rate) | **77.00%** (231/300 queries) |
| Avg Retrieval Time | **35ms** |
| Corpus | SciFact BEIR — 5,183 scientific abstracts |
| Hardware | Intel i3 12th Gen · 8GB RAM · Zorin OS |

## Architecture

```
┌──────────────┐     ┌──────────────────────────────┐
│  User Query  │────▶│  FastEmbed (ONNX, 2 threads)  │
└──────────────┘     │  BAAI/bge-small-en-v1.5       │
                     └────────────┬─────────────────┘
                                  │ embedding
                                  ▼
                     ┌────────────────────────┐
                     │  ChromaDB HNSW Search  │
                     │  (inner product, k=30) │
                     └────────────┬───────────┘
                                  │ top-30 candidates
                                  ▼
                     ┌────────────────────────┐
                     │  BM25 Rerank           │
                     │  (rank_bm25, k=30)     │
                     └────────────┬───────────┘
                                  │ weighted fusion
                                  ▼
                     ┌────────────────────────┐
                     │  Top-K Final Results   │
                     └────────────┬───────────┘
                                  ▼
                     ┌────────────────────────┐
                     │  Qwen 2.5 1.5B         │
                     │  (Ollama, local)       │
                     └────────────────────────┘
```

## Key Optimizations

### 1. Vector-First Hybrid Retrieval
Pure vector search misses exact keywords like medical acronyms (e.g., "PGE2", "SSD"). Instead of running BM25 across the entire 25k-chunk corpus (slow), we use a **two-stage approach**:
1. **Stage 1**: ChromaDB HNSW retrieves top-30 candidates (~30ms)
2. **Stage 2**: BM25 reranks only those 30 docs (<1ms)
3. **Fusion**: Weighted combination (0.3 BM25 / 0.7 Vector) selects the final top-K

This `VectorFirstHybridRetriever` achieves the same accuracy as full hybrid search at **~35ms** instead of ~130ms.

### 2. Why Not Parallel BM25?
We tested running BM25 (25k chunks) and vector search in parallel via `ThreadPoolExecutor`. Python's GIL prevents true parallelism — both compete for the same CPU cores, yielding no speedup. The vector-first reranking approach is fundamentally faster.

### 3. Embedding Model — BGE-Small v1.5
Upgraded from `all-MiniLM-L6-v2` to `BAAI/bge-small-en-v1.5` — a leaderboard champion specifically trained for RAG and scientific retrieval.

### 4. Hardware-Aware P-Core Locking
Restricted FastEmbed ONNX Runtime to `threads=2`, forcing computation onto the two high-speed Performance Cores (P-Cores) of the Intel 12th Gen hybrid architecture. This avoids context-switching overhead with the Efficiency Cores.

### 5. ChromaDB Inner Product Distance
BGE embeddings are normalized, so cosine similarity degenerates to inner product — which is cheaper to compute. Set via `collection_metadata={"hnsw:space": "ip"}`.

### 6. Optimized Chunking
`chunk_size=500`, `chunk_overlap=50` — tuned for SciFact abstracts to reduce total chunks while preserving context boundaries.

### 7. LLM Guardrails

| Parameter | Value | Purpose |
|---|---|---|
| Temperature | 0.2 | Eliminates creative guessing |
| Max Tokens | 256 | Prevents rambling, saves CPU/RAM |
| Prompt | XML Fencing | `<context>` tags for rigid attention boundaries |
| Negative Logic | Strict | Forces "I don't know" when answer is missing |

## Project Structure

```
├── main.py              # Control center menu
├── ingest.py            # Ingests SciFact corpus into ChromaDB
├── query.py             # Interactive Q&A with streaming LLM
├── beir_eval.py         # BEIR benchmark (300 queries)
├── diag.py              # Diagnostic retriever tests
├── test_retriever.py    # Retriever unit tests
├── check_db.py          # Database inspection utility
├── delete.py            # Document pruning tool
├── download_sciFact.py  # Dataset downloader
├── fetch_test.py        # Fetch testing utility
├── scifact/             # SciFact BEIR dataset (not tracked)
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/test.tsv
├── chroma_db/           # ChromaDB vector store (not tracked)
└── rag-env/             # Python venv (not tracked)
```

## Setup

```bash
# Clone
git clone https://github.com/vbg-07/hybrid-rag.git
cd hybrid-rag

# Create environment
python3 -m venv rag-env
source rag-env/bin/activate

# Install dependencies
pip install langchain langchain-community langchain-chroma langchain-ollama
pip install langchain-text-splitters fastembed rank-bm25 numpy chromadb

# Download SciFact dataset
python download_sciFact.py

# Install Ollama + model
# https://ollama.com/download
ollama pull qwen2.5:1.5b

# Ingest corpus
python ingest.py

# Run benchmark
python beir_eval.py

# Interactive Q&A
python query.py

# Or use the control center
python main.py
```

## License

MIT
