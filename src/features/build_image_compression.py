import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def image_compression(image_input, k=6):
    """
    Compress an image using the K-Means clustering algorithm.

    Parameters:
        image_input (str or numpy array or PIL.Image): Path to the input image, an image array, or a PIL image.
        k (int): Number of color clusters (default is 8).

    Returns:
        compressed_image (PIL.Image.Image): The compressed image in PIL format.
    """
    # Check if input is a file path (string), PIL image or numpy array
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    elif isinstance(image_input, Image.Image):  # If input is a PIL image
        image = np.array(image_input)  # Convert PIL image to NumPy array
    else:
        image = image_input  # If input is already a NumPy array

    # Ensure image is in RGB format
    if image.ndim == 3 and image.shape[2] == 4:  # If the image has alpha channel (RGBA)
        image = image[:, :, :3]  # Remove alpha channel to get RGB
    
    # Reshape the image into a 2D array (num_pixels, 3)
    pixels = image.reshape((-1, 3))
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Replace each pixel with the color of its nearest cluster center
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = compressed_pixels.reshape(image.shape).astype(np.uint8)
    
    # Convert the result back to PIL.Image
    compressed_image_pil = Image.fromarray(compressed_image)
    
    return compressed_image_pil