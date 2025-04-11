import cv2
import numpy as np
from PIL import Image

def add_caption_to_image(image_pil, caption: str) -> Image.Image:
    """
    Add a caption below a small PIL image using OpenCV.

    Args:
        image_pil (PIL.Image.Image): Input image (small size).
        caption (str): Text to add as caption below the image.

    Returns:
        PIL.Image.Image: Image with caption added below.
    """
    # Convert PIL to OpenCV (RGB -> BGR)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Image dimensions
    img_height, img_width = image_cv.shape[:2]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, img_width / 1024)     # auto-scale font based on width
    font_thickness = 1

    # Text size
    text_size, _ = cv2.getTextSize(caption, font, font_scale, font_thickness)
    text_width, text_height = text_size

    # Padding
    padding_top = 10
    padding_bottom = 10
    caption_area_height = text_height + padding_top + padding_bottom

    # Create new white canvas
    new_img_height = img_height + caption_area_height
    new_image = np.ones((new_img_height, img_width, 3), dtype=np.uint8) * 255  # white background
    new_image[:img_height, :, :] = image_cv  # paste original image

    # Text position (centered)
    text_x = (img_width - text_width) // 2
    text_y = img_height + padding_top + text_height

    # Draw text
    cv2.putText(new_image, caption, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
