import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import open_clip
import chromadb

# ==========================================
# 1. SET YOUR QUERY HERE
# ==========================================
#QUERY_TYPE = "text"   # Options: "text", "image", "audio"
#QUERY_VALUE = "small bird with black spots, long beak, and colored eyebrows"  # For text, you can also provide a .txt file path

# Examples for other modalities:
QUERY_TYPE = "image"
QUERY_VALUE = "C:\\Users\\ivani\\Desktop\\UPC\\ADSDB\\ADSDB-project-main-YAMNET\\flicker.jpg"  
#
#QUERY_TYPE = "audio"
#QUERY_VALUE = "C:\\Users\\ivani\\Desktop\\UPC\\ADSDB\\ADSDB-project-main-YAMNET\\XC975331 - Northern Flicker - Colaptes auratus.wav"
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "chroma_data"
MULTIMODAL_COLLECTION = "bird_multimodal_clip"
MODEL_NAME = "ViT-L-14"
PRETRAINED = "laion2b_s32b_b82k"


def get_audio_mapper(client):
    """Dynamically trains the W matrix to project Audio into CLIP space."""
    from sklearn.linear_model import Ridge
    
    audio_db = client.get_collection("audio_yamnet")
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

    print(f"\n=========================================")
    print(f" SEARCHING BY {QUERY_TYPE.upper()}:")
    print(f" '{QUERY_VALUE}'")
    print(f"=========================================\n")

    # ------------------------------------------
    # Embed the Query based on the Modality
    # ------------------------------------------
    if QUERY_TYPE in ["text", "image"]:
        print(f"Loading CLIP model ({MODEL_NAME}) on {device}...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=device
        )
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.eval()

        if QUERY_TYPE == "text":
            # Check if the user passed a .txt file path instead of a raw string
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
            # PIL handles .png, .jpg, .jpeg automatically. 
            # .convert("RGB") safely removes alpha channels from PNGs.
            image = Image.open(QUERY_VALUE).convert("RGB")
            with torch.no_grad():
                tensor = preprocess(image).unsqueeze(0).to(device)
                emb = model.encode_image(tensor)
                emb /= emb.norm(dim=-1, keepdim=True)
                query_vector = emb[0].cpu().tolist()

    elif QUERY_TYPE == "audio":
        print(f"Loading YAMNet and processing Audio...")
        import tensorflow_hub as hub
        from sklearn.preprocessing import normalize
        import librosa  # Make sure to run: pip install librosa
        
        # librosa handles .mp3, .wav, .flac natively.
        # sr=16000 and mono=True forces the exact format YAMNet requires!
        waveform, _ = librosa.load(QUERY_VALUE, sr=16000, mono=True)
        
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        _, embeddings, _ = yamnet(waveform)
        
        yamnet_vector = np.asarray(embeddings).mean(axis=0).astype(np.float32)
        yamnet_vector = yamnet_vector / np.linalg.norm(yamnet_vector)
        
        print("Projecting Audio into Multimodal Space...")
        mapper = get_audio_mapper(client)
        projected = mapper.predict([yamnet_vector])
        projected = normalize(projected, norm='l2', axis=1)
        query_vector = projected[0].tolist()

    else:
        print("Invalid QUERY_TYPE. Please choose 'text', 'image', or 'audio'.")
        return

    # ------------------------------------------
    # Retrieve ALL results and extract actual data
    # ------------------------------------------
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

    # ------------------------------------------
    # Print the Results
    # ------------------------------------------
    for modality in ["text", "image", "audio_projected"]:
        display_name = "AUDIO" if modality == "audio_projected" else modality.upper()
        print(f"--- Top 4 {display_name} Results ---")
        
        for i, res in enumerate(top_results[modality]):
            print(f"{i+1}. Distance: {res['distance']:.4f} | Species: {res['species']}")
            
            if modality == "text" and res["document"]:
                snippet = res["document"].replace("\n", " ").strip()
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                print(f"   Description: \"{snippet}\"")
            else:
                print(f"   File: {res['file_key']}")
        print("\n")

if __name__ == "__main__":
    main()