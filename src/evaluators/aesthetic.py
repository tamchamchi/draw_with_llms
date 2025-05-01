import os

import clip
import torch
import torch.nn as nn
from PIL import Image

from configs.configs import MODEL_DIR


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self):
        self.model_path = os.path.join(MODEL_DIR, "sac+logos+ava1-l14-linearMSE.pth")
        self.clip_model_path = os.path.join(MODEL_DIR, "ViT-L-14.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """Loads the aesthetic predictor model and CLIP model."""
        state_dict = torch.load(
            self.model_path, weights_only=True, map_location=self.device
        )

        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to(self.device)
        predictor.eval()
        clip_model, preprocessor = clip.load(self.clip_model_path, device=self.device)

        return predictor, clip_model, preprocessor

    def score(self, image: Image.Image) -> float:
        """Predicts the CLIP aesthetic score of an image."""
        image = self.preprocessor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()

        score = self.predictor(torch.from_numpy(image_features).to(self.device).float())

        return score.item() / 10.0  # scale to [0, 1]

    def compute_clip_similarity(self, image: Image.Image, text: str) -> float:
        """
        Compute the similarity score between an image and a text using the CLIP model.

        Args:
            image (PIL.Image.Image): The input image to be compared.
            text (str): The text description to compare with the image.

        Returns:
            float: A similarity score between the image and text embeddings.
                Higher scores indicate greater similarity.
        """

        # Preprocess the image (resize, normalize, etc.), add batch dimension, and move to the correct device
        image = self.preprocessor(image).unsqueeze(0).to(self.device)

        # Tokenize the input text and move it to the correct device
        text = clip.tokenize([text]).to(self.device)

        # Disable gradient computation for inference
        with torch.no_grad():
            # Extract image features using CLIP
            image_features = self.clip_model.encode_image(image)
            # Extract text features using CLIP
            text_features = self.clip_model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity (dot product since CLIP features are normalized)
        similarity = max((image_features @ text_features.T).item(), 0)

        return similarity
