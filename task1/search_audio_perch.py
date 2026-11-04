import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import chromadb
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path

print("Loading Google Perch v2 model...")
perch_model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/1')
infer_signature = perch_model.signatures['serving_default']

CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma_data"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
audio_collection = chroma_client.get_collection(name="bird_audio_perch")

new_audio_path = sys.argv[1].strip('"')
audio_array, sample_rate = sf.read(new_audio_path)

if audio_array.ndim > 1:
    audio_array = np.mean(audio_array, axis=1)

if sample_rate != 32000:
    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=32000)

chunk_samples = 32000 * 5
chunk_embeddings = []
chunks = []

if len(audio_array) <= chunk_samples:
    padded = np.pad(audio_array, (0, chunk_samples - len(audio_array)), 'constant')
    chunks = [padded]
else:
    for start in range(0, len(audio_array) - chunk_samples + 1, chunk_samples):
        chunks.append(audio_array[start:start + chunk_samples])

print(f"Split audio into {len(chunks)} chunk(s). Extracting and averaging embeddings...")

for chunk in chunks:
    waveform = chunk.astype(np.float32)[np.newaxis, :]
    outputs = infer_signature(inputs=tf.constant(waveform))
    chunk_embeddings.append(outputs['embedding'][0].numpy())

final_vector = np.mean(chunk_embeddings, axis=0).tolist()

results = audio_collection.query(
    query_embeddings=[final_vector],
    n_results=5
)

print("\nTop 5 Matches for Whole Audio")
for i in range(5):
    match_id = results['ids'][0][i]
    match_dist = results['distances'][0][i]
    match_species = results['metadatas'][0][i]['species']
    
    print(f"Rank {i+1}: {match_id}")
    print(f"-> Species: {match_species}")
    print(f"-> Distance: {match_dist:.4f}\n")