from pathlib import Path
from collections import defaultdict

import chromadb
import numpy as np


# Configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "chroma_data"

IMAGE_COLLECTION = "bird_images_clip"
TEXT_COLLECTION = "bird_texts_openclip"
MULTIMODAL_COLLECTION = "bird_multimodal_clip"

START_BIRD = 1
END_BIRD = 100
MAX_PER_SPECIES = 5
EMBEDDING_DIMENSION = 768

# Select up to 5 records per species
def select_records(data, modality):

    selected = []
    counts = defaultdict(int)

    # Sort by source key to ensure deterministic selection
    records = sorted(
        zip(
            data["ids"],
            data["embeddings"],
            data["metadatas"],
            data["documents"]
        ),
        key=lambda record: record[2].get(
            "minio_key",
            record[2].get("source_key", "")
        )
    )

    for record_id, embedding, metadata, document in records:
        species = metadata["species"]

        try:
            species_number = int(species[:3])
        except ValueError:
            continue

        if not START_BIRD <= species_number <= END_BIRD:
            continue

        if counts[species] >= MAX_PER_SPECIES:
            continue

        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Unexpected embedding dimension for {record_id}"
            )

        if metadata.get("model") != "ViT-L-14":
            raise ValueError("Unexpected CLIP model.")

        if modality == "text" and metadata.get(
            "pretrained"
        ) != "laion2b_s32b_b82k":
            raise ValueError("Unexpected text checkpoint.")

        new_metadata = dict(metadata)
        new_metadata["modality"] = modality

        selected.append((
            f"{modality}:{record_id}",
            np.asarray(embedding).tolist(),
            new_metadata,
            document
        ))

        counts[species] += 1

    print(f"\n{modality.upper()} records: {len(selected)}")

    for species, count in sorted(counts.items()):
        print(f"{species}: {count}")

    return selected, set(counts)


def main():
    print("Creating multimodal CLIP collection...")

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    images = client.get_collection(
        name=IMAGE_COLLECTION
    )

    texts = client.get_collection(
        name=TEXT_COLLECTION
    )

    image_data = images.get(
        include=["embeddings", "metadatas"]
    )

    # Images have no stored documents
    image_data["documents"] = [
        None
    ] * len(image_data["ids"])

    text_data = texts.get(
        include=["embeddings", "metadatas", "documents"]
    )

    image_records, image_species = select_records(
        image_data, "image"
    )

    text_records, text_species = select_records(
        text_data, "text"
    )

    if image_species != text_species:
        raise RuntimeError(
            "Image and text species do not match."
        )

    if not image_records or not text_records:
        raise RuntimeError(
            "Missing image or text embeddings."
        )

    collection = client.get_or_create_collection(
        name=MULTIMODAL_COLLECTION,
        configuration={"hnsw": {"space": "cosine"}}
    )

    for modality_records in [image_records, text_records]:

        ids, embeddings, metadatas, documents = zip(
            *modality_records
        )

        # ChromaDB stores image metadata without documents
        if documents[0] is None:
            collection.upsert(
                ids=list(ids),
                embeddings=list(embeddings),
                metadatas=list(metadatas)
            )
        else:
            collection.upsert(
                ids=list(ids),
                embeddings=list(embeddings),
                metadatas=list(metadatas),
                documents=list(documents)
            )

    print("\nMultimodal collection created!")
    print("Collection:", MULTIMODAL_COLLECTION)
    print("Images:", len(image_records))
    print("Texts:", len(text_records))
    print("Total entries:", collection.count())


if __name__ == "__main__":
    main()
