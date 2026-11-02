import numpy as np
import chromadb
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
import re

PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "chroma_data"

AUDIO_COLLECTION = "bird_audio_perch"
MULTIMODAL_COLLECTION = "bird_multimodal_clip"

def clean_species_name(name):
    """Normalize species name by removing leading numbers/dots (e.g. '036.Northern_Flicker' -> 'Northern_Flicker')"""
    if not name:
        return ""
    # clean name
    cleaned = re.sub(r'^\d+\W*', '', name)
    return cleaned.strip().lower()

def main():
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    
    audio_db = client.get_collection(name=AUDIO_COLLECTION)
    mm_db = client.get_collection(name=MULTIMODAL_COLLECTION)
    
    # get all the data
    audio_data = audio_db.get(include=["embeddings", "metadatas"])
    mm_data = mm_db.get(include=["embeddings", "metadatas"])
    
    # group embeddings by species
    audio_by_species = {}
    for i, meta in enumerate(audio_data["metadatas"]):
        raw_species = meta.get("species", "")
        norm_species = clean_species_name(raw_species)
        if not norm_species:
            continue
        if norm_species not in audio_by_species:
            audio_by_species[norm_species] = []
        audio_by_species[norm_species].append(audio_data["embeddings"][i])
        
    clip_by_species = {}
    for i, meta in enumerate(mm_data["metadatas"]):
        raw_species = meta.get("species", "")
        norm_species = clean_species_name(raw_species)
        if not norm_species:
            continue
        if norm_species not in clip_by_species:
            clip_by_species[norm_species] = []
        clip_by_species[norm_species].append(mm_data["embeddings"][i])

    # build the pairs for training the W matrix
    X_train = []
    Y_train = []
    
    print("Aligning spaces by species...")
    for species in audio_by_species:
        if species in clip_by_species:
            audio_centroid = np.mean(audio_by_species[species], axis=0)
            clip_centroid = np.mean(clip_by_species[species], axis=0)
            
            X_train.append(audio_centroid)
            Y_train.append(clip_centroid)
            
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    print(f"Training pairs created for {len(X_train)} species.")

    if len(X_train) == 0:
        print("Error: Still 0 training pairs. Check your collection metadata species fields.")
        return

    #train the W matrix using Ridge regression
    print("Optimizing transformation matrix W...")
    mapper = Ridge(alpha=1.0, fit_intercept=False)
    mapper.fit(X_train, Y_train)

    # apply the mapper to all audio embeddings
    all_audio_embeddings = np.array(audio_data["embeddings"])
    projected_audio = mapper.predict(all_audio_embeddings)
    
    # normalize embeddings
    projected_audio = normalize(projected_audio, norm='l2', axis=1)

    # add the projected audio embeddings to the multimodal collection
    print(f"Upserting {len(projected_audio)} mapped audio embeddings into {MULTIMODAL_COLLECTION}...")
    
    mapped_ids = []
    mapped_metadatas = []
    
    for i, original_id in enumerate(audio_data["ids"]):
        mapped_ids.append(f"audio:{original_id}")
        
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