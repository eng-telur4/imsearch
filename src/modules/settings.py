import os
import keras
import torch
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

_ = load_dotenv(find_dotenv())

# ! load_initial_images()
# ! get_total_pages()
# * ギャラリーに一度に表示する画像数
IMAGES_PER_PAGE = 16
GALLERY_ROWS = 2
GALLERY_COLUMNS = IMAGES_PER_PAGE // GALLERY_ROWS
if IMAGES_PER_PAGE % GALLERY_ROWS:
    GALLERY_COLUMNS += 1
GALLERY_HEIGHT = 380
GALLERY_IMAGE_HEIGHT = GALLERY_HEIGHT // GALLERY_ROWS
SEARCH_IMAGE_HEIGHT = 280

NAME = os.getenv("DB_NAME")
HOST = os.getenv("DB_HOST")
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
DSN = f"dbname={NAME} host={HOST} user={USER} password={PASSWORD}"

login(os.getenv("HF_TOKEN"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "stabilityai/japanese-stable-clip-vit-l-16"
MODEL = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval().to(DEVICE)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_PATH)

MODEL_ENET = keras.applications.EfficientNetB0(include_top=False, pooling="avg")

search_sql = """
SELECT image_id, file_name,
    1 - (embedding <=> %s) AS cosine_similarity,
    'vector' as method
FROM images ORDER BY cosine_similarity
DESC LIMIT %s;
"""
search_sql_clip = """
SELECT image_id, file_name,
    1 - (embedding_clip <=> %s) AS cosine_similarity,
    'vector' as method
FROM images ORDER BY cosine_similarity
DESC LIMIT %s;
"""

load_sql = """
SELECT image_id, file_name FROM images
ORDER BY image_id
DESC LIMIT %s
OFFSET %s;
"""
