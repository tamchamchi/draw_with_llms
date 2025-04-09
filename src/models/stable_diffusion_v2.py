import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import PIL.Image
from typing import Optional

class StableDiffusionV2:
     def __init__(self):
          self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          self.model_path = "stabilityai/stable-diffusion-2-1"
          self.scheduler = DDIMScheduler.from_pretrained(self.model_path, subfolder="scheduler")
          self.pipe = StableDiffusionPipeline.from_pretrained(
               self.model_path,
               scheduler=self.scheduler,
               torch_dtype=torch.float16,
               variant="fp16",
          ).to(self.device)

     def generate(self, prompt: str, negative_prompt: str = "", num_inference_steps: int = 20, guidance_scale: float = 10) -> PIL.Image.Image:
          """Generates an image based on the given prompt using Stable Diffusion v2.

          Parameters
          ----------
          prompt : str
               The text prompt to generate an image from.
          negative_prompt : str, optional
               The text prompt to avoid in the generated image.
          num_inference_steps : int, optional
               The number of denoising steps (default is 20).
          guidance_scale : float, optional
               The scale for classifier-free guidance (default is 10).

          Returns
          -------
          PIL.Image.Image
               The generated image.
          """
          with torch.autocast("cuda"):
               image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
               ).images[0]

          return image

     def save_image(self, image: PIL.Image.Image, path: str):
          """Saves the generated image to the specified path.

          Parameters
          ----------
          image : PIL.Image.Image
               The image to save.
          path : str
               The file path to save the image.
          """
          image.save(path)
          return path

