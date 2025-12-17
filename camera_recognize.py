import cv2
import time
import faiss
import pickle
import numpy as np
import torch
import clip
from PIL import Image

# ==== Load CLIP ====
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ==== Load FAISS index + metadata ====
index = faiss.read_index("albums.index")
with open("albums.pkl", "rb") as f:
    metadata = pickle.load(f)

# ==== Camera ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera error")
    exit()

last_inference = 0
label = "Scanning..."

def recognize_frame(frame, k=3, threshold=0.85):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        emb = model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)

    emb = emb.cpu().numpy().astype("float32")
    scores, indices = index.search(emb, k)

    if scores[0][0] < threshold:
        return "Unknown"

    top = metadata[indices[0][0]]
    return f"{top['artist']} – {top['album']}"

# ==== Main Loop ====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # Run recognition every 1 second
    if now - last_inference > 1.0:
        label = recognize_frame(frame)
        last_inference = now

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Album Recognition (q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
