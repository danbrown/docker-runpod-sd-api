from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
import PIL
import os
import time

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

# models
MODEL_DATA = {
  "runwayml/stable-diffusion-v1-5": {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "slug": "sd-v1-5",
    "precision": "fp16",
    "revision": "fp16",
  },
  "hakurei/waifu-diffusion": {
    "model_id": "hakurei/waifu-diffusion",
    "slug": "waifu-diffusion",
    "precision": "fp16",
    "revision": "fp16",
  },
}

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

# globals
global loaded_models

# utils
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
def loadModel(model_data: str):

  start = time.time()

  loadPipeline = StableDiffusionPipeline

  # scheduler = getScheduler(model_id, DEFAULT_SCHEDULER, False)

  model_id = model_data["model_id"]

  pipe = loadPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if model_data["precision"] == "fp16" else None,
    revision="fp16" if model_data["revision"] == "fp16" else None,
    use_auth_token=HF_AUTH_TOKEN,
    # scheduler=scheduler,
  )

  # save model
  pipe.save_pretrained(f"./{model_data['slug']}")

  load_time = round((time.time() - start) * 1000)
  print(f"Loaded {model_id} in {load_time}ms")

  return pipe.to("cuda") # TODO make this configurable


# # This is called once when the server starts
def init():
  
  # load models
  for model_data in MODEL_DATA:
    model_id = model_data["model_id"]
    loaded_models[model_id] = loadModel(model_data)

  pass

# # This is the main inference function that is called by the server
def inference(model_inputs):
  model_id = model_inputs.get("model_id", None)
  prompt = model_inputs.get("prompt", None)
  negative_prompt = model_inputs.get("negative_prompt", None)
  scheduler_id = model_inputs.get("scheduler_id", DEFAULT_SCHEDULER)
  width = model_inputs.get("width", 512)
  height = model_inputs.get("height", 512)
  num_inference_steps = model_inputs.get("num_inference_steps", 30)
  guidance_scale = model_inputs.get("guidance_scale", 7.5)

  # special properties to trigger pipelines
  init_image = model_inputs.get("init_image", None)
  mask_image = model_inputs.get("mask_image", None)
  controlnet_hint = model_inputs.get("controlnet_hint", None)

  # seed generator
  seed = model_inputs.get("seed", None)
  if seed == None:
      generator = torch.Generator(device=device)
      generator.seed()
  else:
      generator = torch.Generator(device=device).manual_seed(seed)

  # if prompt is not provided, return error
  if prompt == None:
    return {
      "error": {
        "code": "NO_PROMPT",
        "message": "No prompt provided"
      }
    }

  # check if model is valid, if not return error
  if model_id not in MODEL_DATA.keys():
    return {
      "error": {
        "code": "INVALID_MODEL",
        "message": "Invalid model"
      }
    }

  # if model is not loaded, load it
  if model_id not in loaded_models:
    loaded_models[model_id] = loadModel(model_id)


  # get model
  model_data = MODEL_DATA[model_id]

  pipe = StableDiffusionPipeline.from_pretrained(
    model_data["slug"],
    use_auth_token=HF_AUTH_TOKEN,
  )

  pipe = pipe.to("cuda") # TODO make this configurable

  pipe.safety_checker = lambda images, clip_input: (images, False)

  images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
  ).images
      
  # convert to base64
  images_base64 = []
  for image in images:
    images_base64.append("data:image/png;base64," + encodeBase64Image(image))

  # TEMP save images locally for debugging
  for i, image in enumerate(images):
    image.save(f"./{i}.png")

  return {
    "images": images_base64
  }
