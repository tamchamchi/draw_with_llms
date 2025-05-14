import PIL.Image
import torch
from diffusers import DiffusionPipeline


class SDXL_Vector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "stabilityai/stable-diffusion-xl-base-1.0"

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        self.lora_path = (
            "DoctorDiffusion/doctor-diffusion-s-controllable-vector-art-xl-lora"
        )

        self.pipe.load_lora_weights(self.lora_path)

    def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 10,
        seed: int = 42,
    ) -> PIL.Image.Image:
        """Generates an image based on the given prompt using Stable Diffusion v2.

        Parameters
        ----------
        prompt : str
             The text prompt to generate an image from.
        height : int, optional
             Height of the generated image in pixels (default is 512).
        width : int, optional
             Width of the generated image in pixels (default is 512).
        negative_prompt : str, optional
             The text prompt to avoid in the generated image.
        num_inference_steps : int, optional
             The number of denoising steps (default is 20).
        guidance_scale : float, optional
             The scale for classifier-free guidance (default is 10).
        seed : int, optional
             The seed for random generation to ensure reproducibility (default is None, which results in random generation).

        Returns
        -------
        PIL.Image.Image
             The generated image.
        """
        generator = torch.Generator("cuda").manual_seed(seed)
        image = self.pipe(
               prompt=prompt,
               negative_prompt=negative_prompt,
               num_inference_steps=num_inference_steps,
               guidance_scale=guidance_scale,
               width=width,
               height=height,
               generator=generator,
          ).images[0]

        return image

if __name__ == "__main__":
     model = SDXL_Vector()
     image = model.generate("a purple forest at dusk")
     image.save("sdxl_vector.png")
