import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def delete_document(filename):
    print(f"🔄 Connecting to ChromaDB at '{CHROMA_PATH}'...")
    
    # Initialize the same embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
    
    # Connect to the local database
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Fetch all metadata to find the exact file path stored in the DB
    db_data = vectorstore.get(include=["metadatas"])
    unique_sources = set([
        meta.get("source") for meta in db_data["metadatas"] if meta.get("source")
    ])

    # Try to match the user's input to a source path in the DB
    target_source = None
    for source in unique_sources:
        if filename.lower() in source.lower():
            target_source = source
            break
            
    if not target_source:
        print(f"\n❌ Could not find any document matching '{filename}'.")
        print("Currently indexed files:")
        for src in unique_sources:
            print(f"  - {os.path.basename(src)}")
        return

    print(f"\n🗑️  Found match: {target_source}")
    print("Executing deletion...")
    
    # Fetch the exact Chunk IDs tied to this specific source file
    data_to_delete = vectorstore.get(where={"source": target_source})
    ids_to_delete = data_to_delete.get("ids", [])
    
    if ids_to_delete:
        # Delete only those specific chunks
        vectorstore.delete(ids=ids_to_delete)
        print(f"✅ Successfully deleted {len(ids_to_delete)} chunks associated with '{filename}'!")
    else:
        print("⚠️ Found the file, but no chunks were associated with it.")

if __name__ == "__main__":
    print("=" * 40)
    print("   RAG DOCUMENT DELETION TOOL")
    print("=" * 40 + "\n")
    
    file_to_delete = input("Enter the filename to delete (e.g., 'Machine_learning.txt'): ").strip()
    if file_to_delete:
        delete_document(file_to_delete)
    else:
        print("No filename provided. Exiting.")