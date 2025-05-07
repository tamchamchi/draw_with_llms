import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import time


class Gemma2:
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "google/gemma-2-9b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.prompt_template = """Generate SVG code to visually represent the following text description, while respecting the given constraints.
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>"A red circle with a blue square inside"</description>
```svg
<svg viewBox="0 0 384 384" width="384" height="384">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>


Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted and no ellipses.

<description>"{}"</description>
```svg
<svg viewBox="0 0 384 384" width="384" height="384">
"""
        self.default_svg = """<svg viewBox="0 0 384 384" width="384" height="384">
  <rect x="0" y="0" width="384" height="384" fill="white"/>
</svg>"""

    def generate(
        self, description: str, max_new_tokens: int = 3072, seed: int = 42
    ) -> str:
        try:
            prompt = self.prompt_template.format(description)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # 🔁 Đặt seed khác nhau nếu muốn kết quả khác nhau nhưng có thể tái lập
            torch.manual_seed(seed)

            start_time = time.time()

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                )

            elapsed = time.time() - start_time
            if elapsed > 240:
                raise TimeoutError(
                    f"⏱ Generation exceeded 4 minutes ({elapsed:.2f} seconds)"
                )

            output_decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # 🔍 Trích xuất đoạn SVG cuối cùng
            matches = re.findall(
                r"<svg.*?</svg>", output_decoded, re.DOTALL | re.IGNORECASE
            )
            if matches:
                svg = matches[-1]
                return svg
            else:
                return self.default_svg

        except TimeoutError as e:
            print(str(e))
            return self.default_svg

        except Exception as e:
            print("❌ Generation failed:", str(e))
            return self.default_svg


if __name__ == "__main__":
    model = Gemma2()
    for i in range(3):
        print(model.generate("a lighthouse overlooking the ocean", max_new_tokens=3072, seed=42+i))
