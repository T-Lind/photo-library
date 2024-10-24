from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare the image
image_path = r"C:\Users\tenant\PycharmProjects\photo-library\ex-images\image000000.JPG"
image = Image.open(image_path)

# Process the image and generate embeddings
inputs = processor(images=image, return_tensors="pt", padding=True)
image_features = model.get_image_features(**inputs)

# The image embedding is now in image_features
print(image_features.shape)  # Should be (1, 512) for the base model
print(image_features)