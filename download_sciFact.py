import os
from datasets import load_dataset

# 1. Ensure your documents folder exists
os.makedirs("./documents", exist_ok=True)

print("Downloading SciFact corpus from HuggingFace...")
# 2. Load the specific 'corpus' split (the actual scientific papers)
corpus = load_dataset("mteb/scifact", "corpus", split="corpus")

# 3. Grab a random sample of 50 documents to protect your i3 processor
sample_docs = corpus

print("Writing files to disk...")
# 4. Save them as individual text files for your RAG to ingest
for doc in sample_docs:
    filename = f"./documents/scifact_{doc['_id']}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        # We combine the title and the abstract into one messy file
        f.write(f"Title: {doc['title']}\n\n{doc['text']}")

print("✅ Successfully downloaded 50 SciFact documents to ./documents!")