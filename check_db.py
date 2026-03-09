import chromadb

print("Connecting to ChromaDB...")


client = chromadb.PersistentClient(path="./chroma_db")


collection = client.get_collection(name="langchain")


chunk_count = collection.count()

print("-" * 40)
print(f"✅ SANITY CHECK: Your database contains exactly {chunk_count} chunks.")
print("-" * 40)