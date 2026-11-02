import argparse
import os
import json
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import boto3
import chromadb
import numpy as np
import soundfile as sf
import shutil
from PIL import Image
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

DB_PATH = PROJECT_DIR / "chroma_data"
COLLECTION_NAME = "bird_multimodal_clip"  

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
OUTPUT_DIR = PROJECT_DIR / "task3" / "audio_rag_output" / RUN_ID

QUERY_AUDIO = PROJECT_DIR / "task3" / "input_audio" / "query.wav"
QUESTION = "What bird is singing?"

parser = argparse.ArgumentParser()
parser.add_argument("--audio", type=Path, default=QUERY_AUDIO)
QUERY_AUDIO = parser.parse_args().audio.resolve()
if not QUERY_AUDIO.is_file():
    raise FileNotFoundError(f"Put an external recording at {QUERY_AUDIO}, or pass --audio PATH")
QUERY = QUERY_AUDIO.name

TOP_K = 3

client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_collection(COLLECTION_NAME)

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

# Prepare the recording as in Task 2: mono, 32 kHz and 5 seconds.
audio, rate = sf.read(QUERY_AUDIO)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if not len(audio) or not np.isfinite(audio).all():
    raise ValueError("The input audio is empty or contains invalid samples")
if rate != 32000:
    audio = librosa.resample(audio, orig_sr=rate, target_sr=32000)
audio = audio[:160000]
audio = np.pad(audio, (0, 160000 - len(audio))).astype(np.float32)

PERCH_MODEL = "https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1"
print("Loading Perch v2...", flush=True)
model = hub.load(PERCH_MODEL)
outputs = model.signatures["serving_default"](inputs=tf.constant(audio[np.newaxis, :]))
audio_vector = outputs["embedding"][0].numpy().astype(np.float32)

def species_key(name):
    # Match names starting with number and without
    return re.sub(r"^\d+\W*", "", name or "").strip().lower()

def get_perch_audio_mapper(client):
    """Dynamically trains the W matrix to project Perch Audio into CLIP space."""
    audio_db = client.get_collection("bird_audio_perch")
    mm_db = client.get_collection(COLLECTION_NAME)
    
    audio_data = audio_db.get(include=["embeddings", "metadatas"])
    mm_data = mm_db.get(include=["embeddings", "metadatas"])
    
    audio_by_sp = {}
    for i, m in enumerate(audio_data["metadatas"]):
        audio_by_sp.setdefault(species_key(m["species"]), []).append(audio_data["embeddings"][i])
        
    clip_by_sp = {}
    for i, m in enumerate(mm_data["metadatas"]):
        if m.get("modality") in ("text", "image"):
            clip_by_sp.setdefault(species_key(m["species"]), []).append(mm_data["embeddings"][i])

    X_train, Y_train = [], []
    for sp in audio_by_sp:
        if sp in clip_by_sp:
            X_train.append(np.mean(audio_by_sp[sp], axis=0))
            Y_train.append(np.mean(clip_by_sp[sp], axis=0))
            
    mapper = Ridge(alpha=1.0, fit_intercept=False)
    if not X_train:
        raise ValueError("No species shared by the Perch and original CLIP records")
    mapper.fit(np.array(X_train), np.array(Y_train))
    print(f"Projection fitted with {len(X_train)} species.", flush=True)
    return mapper


mapper = get_perch_audio_mapper(client)
# Apply Ridge to the raw Perch vector, then normalize as in map_all_perch.py.
query_vector = normalize(mapper.predict([audio_vector]), norm="l2", axis=1)[0].tolist()
print("Searching ChromaDB...", flush=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = collection.query(
    query_embeddings=[query_vector],
    n_results=collection.count(),
    include=["metadatas", "documents", "distances"]
)

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

    if modality == "text":
        item["description"] = document

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

rag_context = {
    "query": QUERY,
    "query_type": "audio",
    "encoder": PERCH_MODEL,
    "audio_preprocessing": "mono, 32000 Hz, first 5 seconds padded if shorter",
    "question": QUESTION,
    "retrieved": retrieved
}

with open(OUTPUT_DIR / "rag_context.json", "w", encoding="utf-8") as file:
    json.dump(rag_context, file, indent=4, ensure_ascii=False)

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

zip_path = shutil.make_archive(
    str(OUTPUT_DIR),
    "zip",
    root_dir=OUTPUT_DIR
)

print("ZIP created:", zip_path)
