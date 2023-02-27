from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
import PIL
import os
import time

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

# models
MODEL_IDS = [
  "runwayml/stable-diffusion-v1-5",
  "hakurei/waifu-diffusion",
  "runwayml/stable-diffusion-inpainting",
  "stabilityai/stable-diffusion-2"
]

FP16_MODELS = [
  "runwayml/stable-diffusion-v1-5",
  "hakurei/waifu-diffusion",
  "runwayml/stable-diffusion-inpainting",
  "stabilityai/stable-diffusion-2"
]

# schedulers
from diffusers import schedulers as _schedulers
SCHEDULERS = [
    "DPMSolverMultistepScheduler",
    "LMSDiscreteScheduler",
    "DDIMScheduler",
    "PNDMScheduler",
    "EulerAncestralDiscreteScheduler",
    "EulerDiscreteScheduler",
]

DEFAULT_SCHEDULER = os.getenv("DEFAULT_SCHEDULER", SCHEDULERS[0])

def decodeBase64Image(imageStr: str, name: str) -> PIL.Image:
  image = PIL.Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))
  print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
  return image

def encodeBase64Image(image: PIL.Image) -> str:
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  return base64.b64encode(buffered.getvalue()).decode("utf-8")

def getScheduler(model_id: str, scheduler_id: str, load: bool) -> str:
  print(f"Initializing {scheduler_id} for {model_id}...")

  start = time.time()

  scheduler = getattr(_schedulers, scheduler_id)
  if scheduler == None:
    return None

  inittedScheduler = scheduler.from_pretrained(
    model_id=model_id,
    use_auth_token=HF_AUTH_TOKEN,
    subfolder="scheduler",
    local_files_only=not load
  )

  diff = round((time.time() - start) * 1000)
  print(f"Initialized {scheduler_id} for {model_id} in {diff}ms")

  return inittedScheduler

# load model
def loadModel(model_id: str):

  start = time.time()

  loadPipeline = StableDiffusionPipeline

  # scheduler = getScheduler(model_id, DEFAULT_SCHEDULER, False)

  model = loadPipeline.from_pretrained(
    model_id, 
    torch_dtype=None if model_id not in FP16_MODELS else torch.float16,
    use_auth_token=HF_AUTH_TOKEN,
    # scheduler=scheduler,
  )

  load_time = round((time.time() - start) * 1000)
  print(f"Loaded {model_id} in {load_time}ms")

  return model.to("cuda") # TODO make this configurable


# # This is called once when the server starts
def init():
  
  # load models
  models = {}
  for model_id in MODEL_IDS:
    models[model_id] = loadModel(model_id)

  pass

# # This is the main inference function that is called by the server
def inference(model_inputs):
  model_id = model_inputs.get("model_id", MODEL_IDS[0])
  prompt = model_inputs.get("prompt", None)
  negative_prompt = model_inputs.get("negative_prompt", None)
  scheduler_id = model_inputs.get("scheduler_id", DEFAULT_SCHEDULER)
  width = model_inputs.get("width", 512)
  height = model_inputs.get("height", 512)

  if prompt == None:
    return {
      "error": {
        "code": "NO_PROMPT",
        "message": "No prompt provided"
      }
    }

  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=HF_AUTH_TOKEN,
  )

  pipe = pipe.to("cuda") # TODO make this configurable

  pipe.safety_checker = lambda images, clip_input: (images, False)

  images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
  )
      
  # convert to base64
  images = [encodeBase64Image(image) for image in images]

  return {
    "images": images
  }