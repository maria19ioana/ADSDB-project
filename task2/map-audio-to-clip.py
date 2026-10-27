import numpy as np
import chromadb
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

# Configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "chroma_data"

AUDIO_COLLECTION = "audio_yamnet"
MULTIMODAL_COLLECTION = "bird_multimodal_clip"

def main():
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    
    audio_db = client.get_collection(name=AUDIO_COLLECTION)
    mm_db = client.get_collection(name=MULTIMODAL_COLLECTION)
    
    # 1. Fetch all data
    audio_data = audio_db.get(include=["embeddings", "metadatas"])
    mm_data = mm_db.get(include=["embeddings", "metadatas"])
    
    # 2. Group embeddings by species to create anchor points
    # We will use the average (centroid) of the species in each space to train W
    audio_by_species = {}
    for i, meta in enumerate(audio_data["metadatas"]):
        species = meta["species"]
        if species not in audio_by_species:
            audio_by_species[species] = []
        audio_by_species[species].append(audio_data["embeddings"][i])
        
    clip_by_species = {}
    for i, meta in enumerate(mm_data["metadatas"]):
        species = meta["species"]
        if species not in clip_by_species:
            clip_by_species[species] = []
        clip_by_species[species].append(mm_data["embeddings"][i])

    # 3. Build the paired training arrays (X = Audio, Y = CLIP)
    X_train = []
    Y_train = []
    
    print("Aligning spaces by species...")
    for species in audio_by_species:
        if species in clip_by_species:
            # Calculate the centroid for the species in both spaces
            audio_centroid = np.mean(audio_by_species[species], axis=0)
            clip_centroid = np.mean(clip_by_species[species], axis=0)
            
            X_train.append(audio_centroid)
            Y_train.append(clip_centroid)
            
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    print(f"Training pairs created for {len(X_train)} species.")

    # 4. Train the W Matrix (Linear Transformation)
    # Goal: find the W that makes d(audio*W, clip) as small as possible
    # We use Ridge regression (linear algebra) without an intercept to map origins correctly
    print("Optimizing transformation matrix W...")
    mapper = Ridge(alpha=1.0, fit_intercept=False)
    mapper.fit(X_train, Y_train)

    # 5. Apply the W Matrix to all audio embeddings
    all_audio_embeddings = np.array(audio_data["embeddings"])
    projected_audio = mapper.predict(all_audio_embeddings)
    
    # Normalize the new embeddings (crucial for cosine similarity in CLIP)
    projected_audio = normalize(projected_audio, norm='l2', axis=1)

    # 6. Upsert the projected audio into the multimodal collection
    print(f"Upserting {len(projected_audio)} mapped audio embeddings into {MULTIMODAL_COLLECTION}...")
    
    mapped_ids = []
    mapped_metadatas = []
    
    for i, original_id in enumerate(audio_data["ids"]):
        mapped_ids.append(f"audio:{original_id}")
        
        # Tag it so you know it's projected audio
        new_meta = dict(audio_data["metadatas"][i])
        new_meta["modality"] = "audio_projected" 
        mapped_metadatas.append(new_meta)

    mm_db.upsert(
        ids=mapped_ids,
        embeddings=projected_audio.tolist(),
        metadatas=mapped_metadatas
    )

    print("Success! Audio is now in the multimodal space.")
    print(f"Total elements in {MULTIMODAL_COLLECTION}: {mm_db.count()}")

if __name__ == "__main__":
    main()