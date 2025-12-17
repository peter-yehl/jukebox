from PIL import Image
import torch
import clip

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("cover.jpg")).unsqueeze(0)

with torch.no_grad():
    image_features = model.encode_image(image)
