from __future__ import annotations

import os
from typing import Union, Optional
import torch, requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

ImageInput = Union[str, Image.Image]

class FlorenceCaptioner:
    """Caption images with a Florence-2 model via Hugging Face Transformers.

    - Loads model + processor once.
    - Supports file paths or PIL.Image.
    - Provides simple and detailed caption modes via prompt token.
    """

    def __init__(
        self,
        model_id: str = "MiaoshouAI/Florence-2-large-PromptGen-v1.5",
        prompt_simple: str = "<CAPTION>",
        prompt_detailed: str = "<MORE_DETAILED_CAPTION>",
        device: Optional[str] = None,
    ) -> None:
        
        self.model_id = model_id
        self.prompt_simple = prompt_simple
        self.prompt_detailed = prompt_detailed

        # Resolve device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Load model + processor
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def caption(self, image: ImageInput, detailed: bool = False, max_new_tokens: int = 512) -> str:
        img = self._load_image(image)
        prompt = self.prompt_detailed if detailed else self.prompt_simple

        inputs = self.processor(text=prompt, images=img, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        gen_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )
        gen_text = self.processor.batch_decode(gen_ids, skip_special_tokens=False)[0]

        try:
            parsed = self.processor.post_process_generation(
                gen_text, task=prompt, image_size=img.size
            )
            # Try to extract a caption field if provided
            if isinstance(parsed, dict):
                payload = parsed.get(prompt) or parsed
                if isinstance(payload, dict):
                    return (
                        payload.get("caption")
                        or payload.get("text")
                        or payload.get("output")
                        or next(iter(payload.values()), "")
                    )
                return str(payload)
            return str(parsed)
        except Exception:
            # Fallback to raw text if post-processing is unavailable
            return gen_text

    @staticmethod
    def _load_image(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                # Basic URL fetch for convenience
                resp = requests.get(image, stream=True)
                resp.raise_for_status()
                return Image.open(resp.raw).convert("RGB")
            # Local path
            return Image.open(image).convert("RGB")
        raise TypeError("image must be a file path, URL, or PIL.Image.Image")


#TODO - quick example, delete later:
if __name__ == "__main__":
    capper = FlorenceCaptioner()
    demo_url = "https://brownvethospital.com/wp-content/uploads/2023/09/cat-happiness-scaled.jpg"
    print(capper.caption(demo_url, detailed=True))
