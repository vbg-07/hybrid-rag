# Hybrid RAG — High-Performance Local Retrieval-Augmented Generation

A fully local RAG pipeline optimized for scientific document retrieval, benchmarked on the **SciFact BEIR** dataset (300 queries). Achieves **77.33% accuracy** with sub-120ms retrieval on consumer hardware — no GPU or cloud APIs required.

## Performance

| Metric | Value |
|---|---|
| Accuracy (Hit Rate) | **77.33%** (232/300 queries) |
| Avg Retrieval Time | ~116ms (hybrid) / ~50ms (vector-only) |
| Corpus | SciFact BEIR — 5,183 scientific abstracts |
| Hardware | Intel i3 12th Gen · 8GB RAM · Zorin OS |

## Architecture

```
┌──────────────┐     ┌──────────────────────────┐
│  User Query  │────▶│  FastEmbed (ONNX, 2 threads) │
└──────────────┘     │  BAAI/bge-small-en-v1.5  │
                     └────────┬─────────────────┘
                              │ embedding
               ┌──────────────┴──────────────┐
               ▼ (parallel via ThreadPool)   ▼
      ┌────────────────┐          ┌─────────────────┐
      │  BM25 Keyword  │          │  ChromaDB HNSW   │
      │  (rank_bm25)   │          │  (inner product) │
      │  weight: 0.3   │          │  weight: 0.7     │
      └───────┬────────┘          └────────┬─────────┘
              └──────────┬─────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Reciprocal Rank    │
              │  Fusion (RRF)       │
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │  Qwen 2.5 1.5B     │
              │  (Ollama, local)    │
              └─────────────────────┘
```

## Key Optimizations

### 1. Hybrid Retrieval (BM25 + Vector)
Pure vector search misses exact keywords like medical acronyms (e.g., "PGE2", "SSD"). We combine **ChromaDB** (semantic) with **BM25** (keyword) using weighted Reciprocal Rank Fusion at **0.3 / 0.7** weights.

### 2. Parallel Retrieval
Replaced LangChain's sequential `EnsembleRetriever` with a custom `ParallelHybridRetriever` that runs BM25 and vector search concurrently via `ThreadPoolExecutor`.

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
