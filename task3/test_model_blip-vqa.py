from io import BytesIO
import boto3
from PIL import Image
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

# MinIO connection
s3 = boto3.client(
    "s3",
    endpoint_url="http://127.0.0.1:9000",
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
)

# Select the image from the trusted zone bucket
BUCKET = "trustedzone"

## 1st testing : a purple finch with food in its mouth
#IMAGE_PATH = "huggingface_cub200/images/035.Purple_Finch/huggingface_cub200_20261009T172126Z_Purple_Finch_0115_28190.jpg"

## 2nd testing : a gadwall swimming
#IMAGE_PATH = "huggingface_cub200/images/046.Gadwall/huggingface_cub200_20261009T172126Z_Gadwall_0014_31570.jpg"

## 3d testing : a bird sitting on a branch
IMAGE_PATH = "huggingface_cub200/images/014.Indigo_Bunting/huggingface_cub200_20261028T221811Z_Indigo_Bunting_0002_12163.jpg"

# Load image from MinIO
response = s3.get_object(Bucket=BUCKET, Key=IMAGE_PATH)
image = Image.open(BytesIO(response["Body"].read())).convert("RGB")

# Hugging Face BLIP-VQA model
vqa_pipeline = pipeline(
    "visual-question-answering",
    model="Salesforce/blip-vqa-capfilt-large"
)

## Question 1:
#question = "What is the bird having in its mouth?"
## Answer 1: food

## Question 2:
#question = "What is the bird doing?"
## Answer 2: swimming

## Question 3:
question = "Where is the bird sitting ?"
## Answer 3: on branch

result = vqa_pipeline(image, question, top_k=1)

print("Question:", question)
print("Answer:", result[0]["answer"])

image.show()
