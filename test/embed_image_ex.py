from get_emb import get_image_embedding

# Example usage
embedding = get_image_embedding('../ex-images/IMG_8908.jpeg')
print(len(embedding))
print(embedding[:5])