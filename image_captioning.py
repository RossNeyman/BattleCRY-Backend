import torch, requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from transformers import modeling_utils as _hf_mu


model = AutoModelForCausalLM.from_pretrained("MiaoshouAI/Florence-2-large-PromptGen-v1.5", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("MiaoshouAI/Florence-2-large-PromptGen-v1.5", trust_remote_code=True)

# Resolve device and move model
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")
model = model.to(device)

prompt = "<MORE_DETAILED_CAPTION>"

url = "https://brownvethospital.com/wp-content/uploads/2023/09/cat-happiness-scaled.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = processor(text=prompt, images=image, return_tensors="pt")
# BatchFeature doesn't support .to(device) directly; send tensors individually
inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3
)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(
    generated_text, task=prompt, image_size=(image.width, image.height)
)

print(parsed_answer)
