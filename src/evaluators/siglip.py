from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch.nn.functional as F
import torch

class Siglip:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "google/siglip-so400m-patch14-384"
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
    
    def compute_siglip_similarity(self, image: Image.Image, text: str) -> float:
        """
        Compute the similarity score between an image and a text using the SigLIP model.

        Args:
            image (PIL.Image.Image): The input image.
            text (str): The text description.

        Returns:
            float: Cosine similarity between image and text embeddings.
        """
        # Preprocess inputs
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds  # shape: (1, dim)
            text_embeds = outputs.text_embeds    # shape: (1, dim)

        # Normalize and compute cosine similarity
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        similarity = (image_embeds * text_embeds).sum().item()

        return similarity

if __name__ == "__main__":
    image = Image.open("/home/anhndt/draw_with_llms/data/results/version_1_02_sd--no_image_compression/02d892-a purple forest at dusk/no_cap - 0 - 0.4760.png")
    text = "a purple forest at dusk"
    model = Siglip()
    print(model.compute_siglip_similarity(image, text))