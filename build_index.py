import json
import faiss
import numpy as np
import torch
import clip
from PIL import Image

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open("albums.json", "r") as f:
    albums = json.load(f)

embeddings = []
metadata = []

for album in albums:
    image = preprocess(Image.open(album["image"])).unsqueeze(0)

    with torch.no_grad():
        emb = model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)

    embeddings.append(emb.cpu().numpy())
    metadata.append(album)

embeddings = np.vstack(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "albums.index")

import pickle
with open("albums.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("Index built with", len(metadata), "albums")
