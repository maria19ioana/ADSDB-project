
"""Task 2 - Multimodal similarity search using CLIP.

Supports:
1. Text -> Image similarity search
2. Image -> Text similarity search

Uses the existing multimodal ChromaDB collection.
"""

import os
from io import BytesIO
from pathlib import Path

import boto3
import chromadb
import matplotlib.pyplot as plt
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from dotenv import load_dotenv


# Configuration

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"

COLLECTION_NAME = "bird_multimodal_clip"
CHROMA_PATH = PROJECT_DIR / "chroma_data"

TOP_K = 3
BUCKET = os.getenv("TRUSTED_BUCKET", "trustedzone")

# Select available device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# Connect to MinIO

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
)


# Connect to ChromaDB

client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# Generate a normalized CLIP text embedding

def generate_text_embedding(query, model, tokenizer):
    """Convert a text query into a CLIP embedding."""

    tokens = tokenizer([query]).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = F.normalize(embedding, dim=-1)

    return embedding[0].cpu().tolist()


# Generate a normalized CLIP image embedding

def generate_image_embedding(image_path, model, preprocess):
    """Convert a local image into a CLIP embedding."""

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = F.normalize(embedding, dim=-1)

    return embedding[0].cpu().tolist()


# Search embeddings by modality

def search_modality(embedding, modality):
    """Retrieve the most similar items from a given modality."""

    return collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        where={"modality": modality},
        include=["metadatas", "documents", "distances"]
    )


# Read an image from MinIO

def load_image(key):
    """Download an image from the Trusted Zone."""

    response = s3.get_object(
        Bucket=BUCKET,
        Key=key
    )

    image_bytes = response["Body"].read()

    return Image.open(BytesIO(image_bytes)).convert("RGB")


# Display images retrieved from a text query

def display_image_results(results):
    """Show the Top 3 images returned by the search."""

    print("\n========== IMAGE RESULTS ==========")

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not metadatas:
        print("No matching images found.")
        return

    fig, axes = plt.subplots(
        1,
        len(metadatas),
        figsize=(13, 4),
        squeeze=False
    )

    axes = axes.flatten()

    for i, (metadata, distance) in enumerate(
        zip(metadatas, distances)
    ):
        image = load_image(metadata["minio_key"])

        species = metadata["species"].split(".", 1)[-1]
        species = species.replace("_", " ")

        similarity = 1 - distance

        axes[i].imshow(image)
        axes[i].set_title(
            f"Rank {i + 1}: {species}\n"
            f"Similarity: {similarity:.3f}",
            fontsize=10
        )
        axes[i].axis("off")

        print(
            f"{i + 1}. {species} | "
            f"Similarity: {similarity:.3f}"
        )

    plt.tight_layout()

    plt.savefig(
        PROJECT_DIR / "task2_similarity_results.png",
        dpi=150
    )

    plt.show()
    plt.close(fig)


# Display descriptions retrieved from an image query

def display_text_results(results):
    """Print the Top 3 text descriptions returned by the search."""

    print("\n========== TEXT RESULTS ==========")

    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    if not metadatas:
        print("No matching descriptions found.")
        return

    for i, (metadata, description, distance) in enumerate(
        zip(metadatas, documents, distances),
        start=1
    ):
        species = metadata["species"].split(".", 1)[-1]
        species = species.replace("_", " ")

        similarity = 1 - distance

        print(f"\nRank {i}")
        print("Species:", species)
        print("Similarity:", round(similarity, 4))
        print("Original MinIO file:", metadata.get("source_key"))
        print("Description:")
        print(description)
        print("-" * 50)


# Main search application

def main():

    print("===================================")
    print(" TASK 2 - MULTIMODAL CLIP SEARCH")
    print("===================================")

    print(f"Using device: {DEVICE}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored embeddings: {collection.count()}")

    print("\nLoading CLIP model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model.eval()

    while True:

        print("\nChoose search type:")
        print("1 - Text -> Image")
        print("2 - Image -> Text")
        print("3 - Text -> Images + Texts (Mixed Search)")
        print("exit - Stop")

        choice = input("Your choice: ").strip().lower()

        if choice == "exit":
            print("\nSearch finished.")
            break

        if choice == "1":

            query = input(
                "Enter a bird description: "
            ).strip()

            if not query:
                print("Please enter a description.")
                continue

            query_embedding = generate_text_embedding(
                query,
                model,
                tokenizer
            )

            image_results = search_modality(
                query_embedding,
                "image"
            )

            display_image_results(image_results)

        elif choice == "2":

            image_path = input(
                "Enter image path: "
            ).strip().strip('"')

            if not Path(image_path).is_file():
                print("Image file not found.")
                continue

            query_embedding = generate_image_embedding(
                image_path,
                model,
                preprocess
            )

            text_results = search_modality(
                query_embedding,
                "text"
            )

            display_text_results(text_results)

        
        elif choice == "3":

            query = input(
                "Enter a bird description: "
            ).strip()

            if not query:
                print("Please enter a description.")
                continue

            # Generate CLIP text embedding
            query_embedding = generate_text_embedding(
                query,
                model,
                tokenizer
            )

            # Retrieve images
            image_results = search_modality(
                query_embedding,
                "image"
            )

            # Retrieve text descriptions
            text_results = search_modality(
                query_embedding,
                "text"
            )

            print("\n===== MIXED SEARCH RESULTS =====")

            display_image_results(image_results)
            display_text_results(text_results)


        else:
            print("Invalid choice. Choose 1, 2 or exit.")


if __name__ == "__main__":
    main()
