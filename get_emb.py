import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch16"


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()

model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def get_image_embeddings(image_paths):
    """Embed a batch of images in a single forward pass.

    Returns a list of embedding lists, in the same order as image_paths.
    """
    images = []
    try:
        for path in image_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))

        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            embeddings = model.get_image_features(**inputs)
        return embeddings.cpu().numpy().tolist()
    finally:
        for img in images:
            img.close()


def get_image_embedding(image_path):
    return get_image_embeddings([image_path])[0]


def get_text_embedding(query):
    inputs = processor(text=query, return_tensors="pt", padding=True).to(DEVICE)
    with torch.inference_mode():
        embeddings = model.get_text_features(**inputs)
    return embeddings.cpu().numpy().flatten().tolist()
