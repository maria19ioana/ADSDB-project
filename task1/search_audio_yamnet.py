"""Task 1: search the three most similar indexed audio recordings."""
import argparse
from pathlib import Path

import chromadb

CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_data"

def collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection("audio_yamnet", embedding_function=None)


def search(xeno_canto_id):
    db = collection()
    query = db.get(where={"xeno_canto_id": xeno_canto_id}, include=["embeddings", "metadatas"])
    if len(query["ids"]) != 1:
        raise ValueError("Expected exactly one indexed recording with that xeno_canto_id")
    metadata = query["metadatas"][0]
    print(f"Query: XC{xeno_canto_id} | {metadata['species']}")
    result = db.query(
        query_embeddings=[query["embeddings"][0].tolist()],
        n_results=min(4, db.count()), include=["metadatas", "distances"],
    )
    print(f"{'Rank':<6}{'Xeno-canto ID':<16}{'Species':<45}Cosine similarity")
    rank = 0
    for key, item, distance in zip(result["ids"][0], result["metadatas"][0], result["distances"][0]):
        if key == query["ids"][0]:
            continue
        rank += 1
        print(f"{rank:<6}{item['xeno_canto_id']:<16}{item['species']:<45}{1 - distance:.4f}")
        print(f"      s3://{item['audio_bucket']}/{item['audio_key']}")
        if rank == 3:
            break
    if rank == 0:
        print("No other recordings indexed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xeno_canto_id")
    args = parser.parse_args()
    search(args.xeno_canto_id)


if __name__ == "__main__":
    main()
