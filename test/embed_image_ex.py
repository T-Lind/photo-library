import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input

# Load the pretrained ResNet model with average pooling
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to preprocess the image
def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return preprocess_input(img_array)

# Extract embeddings
def get_embedding(image_path):
    processed_image = load_and_preprocess_image(image_path)
    embedding = model.predict(processed_image)
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)  # L2 normalization
    return embedding

# Example usage
embedding = get_embedding('../ex-images/image000000.JPG')
print(embedding.shape)  # Should output (1, 2048) for ResNet50 with avg pooling
print(embedding.tolist()[0])