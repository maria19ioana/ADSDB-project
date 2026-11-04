# ADSDB Project

## Environment

Create a `.env` file in the repository root:

```dotenv
MINIO_ENDPOINT=http://127.0.0.1:9000
MINIO_ACCESS_KEY=your_minio_username
MINIO_SECRET_KEY=your_minio_password
XENO_CANTO_API_KEY=your_xeno_canto_api_key
```

## Run the pipeline

From the repository root, use a Python 3.13 environment with FFmpeg on your PATH and MinIO running on port 9000.

```powershell
python -m pip install -r requirements.txt
python -c "from dotenv import load_dotenv; from master_script import main; load_dotenv('.env'); main()"
```

## Run Task 3

After the pipeline finishes, run one of these commands locally:

```powershell
python task3/rag_retrieval.py --query "A dark brown seabird with long wings and a curved beak"
python task3/rag_audio_retrieval.py --audio "C:/path/to/recording.wav"
```

Open [task3_Llava.ipynb](task3/task3_Llava.ipynb) for text or [audio_input_exploration.ipynb](task3/audio_input_exploration.ipynb) for audio in Google Colab. Select a GPU runtime, upload the generated ZIP in the first cell and run the remaining cells. In the text notebook, choose one prompt option before running the generation cell.
