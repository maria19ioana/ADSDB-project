import os
import json
from io import BytesIO
from pathlib import Path

import boto3
import chromadb
import open_clip
import torch
import shutil
from PIL import Image
from dotenv import load_dotenv

# Configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

DB_PATH = PROJECT_DIR / "chroma_data"
COLLECTION_NAME = "bird_multimodal_clip"  # multimodal collection generated in task 2

OUTPUT_DIR = PROJECT_DIR / "task3" / "rag_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY = "A dark brown seabird with long wings and a curved beak"
QUESTION = "What bird species is this, and what are its characteristics?"

TOP_K = 3

# Connect to ChromaDB
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection(COLLECTION_NAME)

# Connect to MinIO
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
)

# Load OpenCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="laion2b_s32b_b82k",
    device=device
)

tokenizer = open_clip.get_tokenizer("ViT-L-14")
model.eval()

# Generate query embedding
tokens = tokenizer([QUERY]).to(device)

with torch.no_grad():
    embedding = model.encode_text(tokens)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

query_vector = embedding[0].cpu().tolist()

# Search the multimodal collection from Task 2
results = collection.query(
    query_embeddings=[query_vector],
    n_results=collection.count(),
    include=["metadatas", "documents", "distances"]
)

# Group results by modality
retrieved = {
    "text": [],
    "image": [],
    "audio_projected": []
}

for record_id, metadata, document, distance in zip(
    results["ids"][0],
    results["metadatas"][0],
    results["documents"][0],
    results["distances"][0]
):
    modality = metadata.get("modality")

    if modality not in retrieved:
        continue

    if len(retrieved[modality]) >= TOP_K:
        continue

    item = {
        "id": record_id,
        "species": metadata.get("species"),
        "distance": distance
    }

    # Collect the actual data associated with the records retrieved from ChromaDB
    # Text: the description is already stored in ChromaDB
    if modality == "text":
        item["description"] = document

    # Image: the original image must be downloaded from MinIO bucket trusted zone
    elif modality == "image":
        key = metadata.get("minio_key")
        bucket = metadata.get("minio_bucket", "trustedzone")

        if key:
            response = s3.get_object(Bucket=bucket, Key=key)
            image = Image.open(
                BytesIO(response["Body"].read())
            ).convert("RGB")

            filename = f"retrieved_image_{len(retrieved['image']) + 1}.jpg"
            image.save(OUTPUT_DIR / filename)

            item["image_file"] = filename
            item["minio_key"] = key

    # Audio: the original audio must also be downloaded from MinIO bucket trusted zone
    elif modality == "audio_projected":
        key = metadata.get("minio_key")
        bucket = metadata.get("minio_bucket", "trustedzone")

        if key:
            local_filename = (
                f"retrieved_audio_{len(retrieved['audio_projected']) + 1}.wav"
            )

            s3.download_file(
                bucket,
                key,
                str(OUTPUT_DIR / local_filename)
            )

            item["audio_file"] = local_filename
            item["minio_key"] = key

    retrieved[modality].append(item)

# Save context for LLaVA
rag_context = {
    "query": QUERY,
    "question": QUESTION,
    "retrieved": retrieved
}

with open(OUTPUT_DIR / "rag_context.json", "w", encoding="utf-8") as file:
    json.dump(rag_context, file, indent=4, ensure_ascii=False)

# Print summary
print("\nRAG retrieval completed!")
print("Output directory:", OUTPUT_DIR)

for modality, items in retrieved.items():
    print(f"\n{modality.upper()} RESULTS:")

    for item in items:
        print("Species:", item["species"])
        print("Distance:", round(item["distance"], 4))

        if modality == "text":
            print("Description:", item.get("description", "")[:200])

        elif modality == "image":
            print("Image:", item.get("image_file"))

        elif modality == "audio_projected":

            print("Audio file:", item.get("audio_file"))
            print("MinIO key:", item.get("minio_key"))

        print("-" * 40)

# Create zip file with the results to upload it in LLaVA
zip_path = shutil.make_archive(
    str(OUTPUT_DIR),
    "zip",
    root_dir=OUTPUT_DIR
)

print("ZIP created:", zip_path)