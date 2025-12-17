import faiss
import pickle
import numpy as np
import torch
import clip
from PIL import Image

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

index = faiss.read_index("albums.index")
with open("albums.pkl", "rb") as f:
    metadata = pickle.load(f)

def recognize(image_path, k=3, threshold=0.85):
    image = preprocess(Image.open(image_path)).unsqueeze(0)

    with torch.no_grad():
        emb = model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)

    emb = emb.cpu().numpy().astype("float32")

    scores, indices = index.search(emb, k)

    # Threshold check using best match
    if scores[0][0] < threshold:
        print("Unknown album")
        return

    for score, idx in zip(scores[0], indices[0]):
        album = metadata[idx]
        print(f"{album['artist']}, {album['album']} ({score:.2f})")

recognize("swimming.jpg")
