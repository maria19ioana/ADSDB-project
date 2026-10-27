import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import open_clip
import chromadb


#QUERY_TYPE = "text"   # Options: "text", "image", "audio"
#QUERY_VALUE = "small bird with black spots, long beak, and colored eyebrows"  

QUERY_TYPE = "image"
QUERY_VALUE = "C:\\Users\\ivani\\Desktop\\UPC\\ADSDB\\ADSDB-fresh-clone\\flicker.jpg"  

#QUERY_TYPE = "audio"
#QUERY_VALUE = "C:\\Users\\ivani\\Desktop\\UPC\\ADSDB\\ADSDB-project-main-YAMNET\\XC975331 - Northern Flicker - Colaptes auratus.wav"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "chroma_data"
MULTIMODAL_COLLECTION = "bird_multimodal_clip"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"


def get_perch_audio_mapper(client):
    """Dynamically trains the W matrix to project Perch Audio into CLIP space."""
    from sklearn.linear_model import Ridge
    
    audio_db = client.get_collection("bird_audio_perch")
    mm_db = client.get_collection(MULTIMODAL_COLLECTION)
    
    audio_data = audio_db.get(include=["embeddings", "metadatas"])
    mm_data = mm_db.get(include=["embeddings", "metadatas"])
    
    audio_by_sp = {}
    for i, m in enumerate(audio_data["metadatas"]):
        audio_by_sp.setdefault(m["species"], []).append(audio_data["embeddings"][i])
        
    clip_by_sp = {}
    for i, m in enumerate(mm_data["metadatas"]):
        clip_by_sp.setdefault(m["species"], []).append(mm_data["embeddings"][i])

    X_train, Y_train = [], []
    for sp in audio_by_sp:
        if sp in clip_by_sp:
            X_train.append(np.mean(audio_by_sp[sp], axis=0))
            Y_train.append(np.mean(clip_by_sp[sp], axis=0))
            
    mapper = Ridge(alpha=1.0, fit_intercept=False)
    mapper.fit(np.array(X_train), np.array(Y_train))
    return mapper


def main():
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=MULTIMODAL_COLLECTION)
    
    total_docs = collection.count()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_vector = None

    print(f" SEARCHING BY {QUERY_TYPE.upper()}:")
    print(f" '{QUERY_VALUE}'")

    if QUERY_TYPE in ["text", "image"]:
        print(f"Loading CLIP model ({MODEL_NAME}) on {device}...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=device
        )
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.eval()

        if QUERY_TYPE == "text":
            if os.path.isfile(QUERY_VALUE) and str(QUERY_VALUE).lower().endswith(".txt"):
                with open(QUERY_VALUE, "r", encoding="utf-8") as f:
                    query_text = f.read().strip()
            else:
                query_text = str(QUERY_VALUE)
                
            with torch.no_grad():
                tokens = tokenizer([query_text]).to(device)
                emb = model.encode_text(tokens)
                emb /= emb.norm(dim=-1, keepdim=True)
                query_vector = emb[0].cpu().tolist()

        elif QUERY_TYPE == "image":
            image = Image.open(QUERY_VALUE).convert("RGB")
            with torch.no_grad():
                tensor = preprocess(image).unsqueeze(0).to(device)
                emb = model.encode_image(tensor)
                emb /= emb.norm(dim=-1, keepdim=True)
                query_vector = emb[0].cpu().tolist()

    elif QUERY_TYPE == "audio":
        print(f"Loading Google Perch v2 and processing Audio...")
        import tensorflow_hub as hub
        import soundfile as sf
        import tensorflow as tf
        from sklearn.preprocessing import normalize
        import librosa 
        
        audio_array, sample_rate = sf.read(QUERY_VALUE)
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        if sample_rate != 32000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=32000)
            
        chunk_samples = 32000 * 5
        if len(audio_array) < chunk_samples:
            audio_array = np.pad(audio_array, (0, chunk_samples - len(audio_array)), 'constant')
        else:
            audio_array = audio_array[:chunk_samples]
            
        waveform = audio_array.astype(np.float32)[np.newaxis, :]
        
        perch_model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1')
        infer_signature = perch_model.signatures['serving_default']
        
        outputs = infer_signature(inputs=tf.constant(waveform))
        perch_vector = outputs['embedding'][0].numpy().astype(np.float32)
        perch_vector = perch_vector / np.linalg.norm(perch_vector)
        
        print("Projecting Perch Audio into Multimodal Space...")
        mapper = get_perch_audio_mapper(client)
        projected = mapper.predict([perch_vector])
        projected = normalize(projected, norm='l2', axis=1)
        query_vector = projected[0].tolist()

    else:
        print("Invalid QUERY_TYPE. Please choose 'text', 'image', or 'audio'.")
        return

    print(f"\nRetrieving and sorting all {total_docs} records...")
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=total_docs,
        include=["metadatas", "distances", "documents"] 
    )

    top_results = {"text": [], "image": [], "audio_projected": []}

    for doc_id, distance, meta, document in zip(
        results["ids"][0], 
        results["distances"][0], 
        results["metadatas"][0], 
        results["documents"][0]
    ):
        modality = meta.get("modality", "")
            
        if modality in top_results and len(top_results[modality]) < 4:
            top_results[modality].append({
                "distance": distance,
                "species": meta.get("species", "unknown"),
                "document": document,
                "file_key": meta.get("minio_key") or meta.get("audio_key", doc_id)
            })

    for modality in ["text", "image", "audio_projected"]:
        display_name = "AUDIO" if modality == "audio_projected" else modality.upper()
        print(f"--- Top 4 {display_name} Results ---")
        
        for i, res in enumerate(top_results[modality]):
            print(f"{i+1}. Distance: {res['distance']:.4f} | Species: {res['species']}")
            
            if modality == "text" and res["document"]:
                snippet = res["document"].replace("\n", " ").strip()
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                print(f"    Description: \"{snippet}\"")
            else:
                print(f"    File: {res['file_key']}")
        print("\n")

if __name__ == "__main__":
    main()