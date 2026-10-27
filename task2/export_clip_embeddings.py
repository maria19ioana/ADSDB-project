
"""Export CLIP embeddings and metadata for audio alignment."""

from pathlib import Path
import csv

import chromadb
import numpy as np


# Configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "chroma_data"

COLLECTION_NAME = "bird_multimodal_clip"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

OUTPUT_DIR = PROJECT_DIR / "task2" / "exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Exporting CLIP embeddings...")

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    data = collection.get(
        include=["embeddings", "metadatas"]
    )

    if not data["ids"]:
        raise RuntimeError("The collection is empty.")

    # Keep a deterministic order
    records = sorted(
        zip(
            data["ids"],
            data["embeddings"],
            data["metadatas"]
        ),
        key=lambda record: record[0]
    )

    ids = []
    embeddings = []
    metadata_rows = []

    for record_id, embedding, metadata in records:
        species = metadata["species"]
        modality = metadata["modality"]

        source_key = metadata.get(
            "minio_key",
            metadata.get("source_key", "")
        )

        ids.append(record_id)
        embeddings.append(embedding)

        metadata_rows.append({
            "id": record_id,
            "species": species,
            "modality": modality,
            "source_key": source_key,
            "model": MODEL_NAME,
            "pretrained": PRETRAINED
        })

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.shape[1] != 768:
        raise RuntimeError(
            f"Unexpected embedding dimension: {embeddings.shape[1]}"
        )

    # Save vectors and matching identifiers
    np.savez_compressed(
        OUTPUT_DIR / "clip_embeddings.npz",
        embeddings=embeddings,
        ids=np.asarray(ids)
    )

    # Save metadata in the same order
    csv_path = OUTPUT_DIR / "clip_metadata.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "species",
                "modality",
                "source_key",
                "model",
                "pretrained"
            ]
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    image_count = sum(
        row["modality"] == "image"
        for row in metadata_rows
    )

    text_count = sum(
        row["modality"] == "text"
        for row in metadata_rows
    )

    print("\nExport completed!")
    print("Images:", image_count)
    print("Texts:", text_count)
    print("Total embeddings:", len(ids))
    print("Embedding shape:", embeddings.shape)
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
