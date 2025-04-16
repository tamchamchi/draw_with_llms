from typing import List

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def add_caption_to_image(image_pil: Image.Image, caption: List[str]) -> Image.Image:
    """
    Add multiple lines of caption below a small PIL image using OpenCV.

    Args:
        image_pil (PIL.Image.Image): Input image (small size).
        caption (List[str]): List of text lines to add as caption, each on a new line.

    Returns:
        PIL.Image.Image: Image with multi-line caption added below.
    """
    # Convert PIL to OpenCV (RGB -> BGR)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Image dimensions
    img_height, img_width = image_cv.shape[:2]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, img_width / 1800)  # auto-scale font based on width
    font_thickness = 1
    line_spacing = 8  # pixels between lines

    # Calculate total caption area height
    text_sizes = [
        cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in caption
    ]
    text_heights = [h for (_, h) in text_sizes]
    total_text_height = sum(text_heights) + line_spacing * (len(caption) - 1)
    padding_top = 10
    padding_bottom = 10
    caption_area_height = total_text_height + padding_top + padding_bottom

    # Create new white canvas
    new_img_height = img_height + caption_area_height
    new_image = (
        np.ones((new_img_height, img_width, 3), dtype=np.uint8) * 255
    )  # white background
    new_image[:img_height, :, :] = image_cv  # paste original image

    # Draw each line of text
    y = img_height + padding_top
    for i, (line, (text_width, text_height)) in enumerate(zip(caption, text_sizes)):
        x = (img_width - text_width) // 2
        cv2.putText(
            new_image,
            line,
            (x, y + text_height),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )
        y += text_height + line_spacing

    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))


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
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):  # If input is a PIL image
        image = np.array(image_input)  # Convert PIL image to NumPy array
    else:
        image = image_input  # If input is already a NumPy array

    # Ensure image is in RGB format
    # If the image has alpha channel (RGBA)
    if image.ndim == 3 and image.shape[2] == 4:
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
