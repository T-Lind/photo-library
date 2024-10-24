from transformers import CLIPProcessor, CLIPModel
import torch
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

image_data = [
    {"path": "../ex-images/image000000.JPG"},
]

import numpy as np
from PIL import Image

def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.cpu().numpy()

# Create embeddings for all images
for item in image_data:
    item['img_embedding'] = get_image_embedding(item['path'])

print(image_data[0]['img_embedding'].shape)  # Output: (1, 512)
# convert into a 1D list of floats
print(len(image_data[0]['img_embedding'].flatten().tolist()))
print(image_data[0]['img_embedding'].flatten().tolist())