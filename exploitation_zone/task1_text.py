
"""Task 1 - Same-modality similarity search using BGE

This script:
1. Accepts a bird description from the user
2. Converts the query into a BGE embedding
3. Searches the existing ChromaDB collection
4. Displays the five most similar descriptions

"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer



# Locate the existing ChromaDB database
PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "chroma_db"

# Use the same model as the embedding pipeline
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Search the BGE text collection
COLLECTION_NAME = "bird_texts_bge"

# Number of similar descriptions to retrieve
TOP_K = 5


# Connect to ChromaDB

def connect_chromadb():
    """Open the existing BGE collection."""

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # get_collection() requires the collection to exist
    # does not create a new empty collection
    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# Search for similar descriptions

def search_similar_text(query, model, collection):
    """Find the most semantically similar bird descriptions."""

    # Convert the user query into a BGE embedding
    # Use the same normalization as embed_text_bge.py
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Search ChromaDB using the stored BGE embeddings
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results


# Display the search results


def display_results(results):
    """Print matching bird descriptions in ranked order."""

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        print("No matching descriptions found.")
        return

    print("\n========== SEARCH RESULTS ==========")

    for index, record_id in enumerate(ids):

        metadata = metadatas[index]
        description = documents[index]
        distance = distances[index]

        # ChromaDB collection uses cosine distance:
        # cosine similarity = 1 - cosine distance
        similarity = 1 - distance

        print(f"\nResult {index + 1}")
        print("-" * 45)

        print("Species:", metadata.get("species", "Unknown"))

        print("Similarity score:", round(similarity, 4))

        print("Cosine distance:", round(distance, 4))

        print("Original MinIO file:",
              metadata.get("source_key", "Unknown"))

        print("Description:")
        print(description)

    print("\n====================================")


def main():

    print("===================================")
    print(" TASK 1 - BGE TEXT SIMILARITY SEARCH")
    print("===================================")

    # Connect to the existing database
    collection = connect_chromadb()

    if collection.count() == 0:
        print("The BGE collection is empty.")
        return

    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Stored embeddings: {collection.count()}")

    # Load the same model used for stored embeddings
    model = SentenceTransformer(MODEL_NAME)

    print("\nEnter a bird description to search.")
    print("Type 'exit' to stop.\n")

    # Allow multiple searches without restarting the script
    while True:

        query = input("Your query: ").strip()

        if query.lower() == "exit":
            print("\nSearch finished.")
            break

        if not query:
            print("Please enter a description.")
            continue

        # Search the existing collection
        results = search_similar_text(
            query,
            model,
            collection
        )

        # Show the matching descriptions
        display_results(results)

        print("\nEnter another query or type 'exit'.")


if __name__ == "__main__":
    main()


# Example Queries
# Query 1: A small bird with a yellow belly and black wings
# Query 2: A bird with a long pointed beak and a brown body
# Query 3: A bird with spots