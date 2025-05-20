from PIL import Image
import vtracer

from src.data.image_processor import svg_to_png

from ..base import ImageProcessingStrategy


class VtracerCompressionStrategy(ImageProcessingStrategy):
    "Vtracer Image Compression Strategy"

    def __init__(self, k: int = 6) -> None:
        print(f"--- Strategy: Init CompressionStrategy with k={k} ---")
        if k <= 0:
            raise ValueError("Parameter k for compression must be positive.")
        self.k = k

    def process(self, image: Image.Image) -> Image.Image:
        print("--- Strategy: Vtracer ---")
        img = image.convert("RGBA")
        resized_img = img.resize((384, 384), Image.Resampling.LANCZOS)
        pixels: list[tuple[int, int, int, int]] = list(resized_img.getdata())

        svg_str: str = vtracer.convert_pixels_to_svg(
            rgba_pixels=pixels,
            size=resized_img.size,
            colormode="color",        # ["color"] or "binary"
            hierarchical="stacked",     # ["stacked"] or "cutout"
            mode="none",             # ["spline"], "polygon", "none"
            filter_speckle=30,   # default: 4
            color_precision=6,  # default: 6
            layer_difference=1, # default: 16
            corner_threshold=60, # default: 60   
            length_threshold=4.0, # in [3.5, 10] default: 4.0
            max_iterations=10,   # default: 10
            splice_threshold=45, # default: 45
            path_precision=8,   # default: 8
        )

        return svg_to_png(svg_str, size=(384, 384))
