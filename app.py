from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
import torch
import base64
from io import BytesIO
import PIL
import os
import time

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

# models
MODEL_DATA = {
  "runwayml/stable-diffusion-v1-5": {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "slug": "sd-v1-5",
    "precision": "fp16",
    "revision": "fp16",
    "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT"]
  },
  "hakurei/waifu-diffusion": {
    "model_id": "hakurei/waifu-diffusion",
    "slug": "waifu-diffusion",
    "precision": "fp16",
    "revision": "fp16",
    "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT"]
  },
}

# pipelines
PIPELINES = [
    "TXT2IMG",
    "IMG2IMG",
    "INPAINT",
    "CONTROLNET",
    "PIX2PIX",
]

DEFAULT_PIPELINE = os.getenv("DEFAULT_PIPELINE", PIPELINES[0])

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

# vars
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_models = {}

# utils
def decodeBase64Image(imageStr: str, name: str) -> PIL.Image:
  image = PIL.Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))
  print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
  return image

def encodeBase64Image(image: PIL.Image) -> str:
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  return base64.b64encode(buffered.getvalue()).decode("utf-8")

def getScheduler(model_id: str, scheduler_id: str) -> str:
  print(f"Initializing {scheduler_id} for {model_id}...")

  start = time.time()

  scheduler = getattr(_schedulers, scheduler_id)
  if scheduler == None:
    raise Exception(f"Scheduler {scheduler_id} not found")

  inittedScheduler = scheduler.from_pretrained(
    model_id,
    use_auth_token=HF_AUTH_TOKEN,
    subfolder="scheduler",
  )

  diff = round((time.time() - start) * 1000)
  print(f"Initialized {scheduler_id} for {model_id} in {diff}ms")

  return inittedScheduler

# load model
def loadModel(model_data: str):

  # create models folder if it does not exist
  if not os.path.exists(PROJECT_PATH + "/models"):
    os.mkdir(PROJECT_PATH + "/models")
  
  model_folder_exists = os.path.exists(PROJECT_PATH + "/models/" + model_data['slug'])

  start = time.time()

  model_id = model_data["model_id"]

  loadPipeline = StableDiffusionPipeline

  # load the model, if is already saved, load from folder, otherwise load from huggingface
  print(f"Loading {model_id} from {'pre_saved' if model_folder_exists else 'huggingface'}...")
  pipe = loadPipeline.from_pretrained(
    model_id if not model_folder_exists else (PROJECT_PATH + "/models/" + model_data['slug']),
    torch_dtype=torch.float16 if model_data["precision"] == "fp16" else None,
    revision="fp16" if model_data["revision"] == "fp16" else None,
    use_auth_token=HF_AUTH_TOKEN,
  )
  
  # save model if it is not already saved
  if not model_folder_exists:
    pipe.save_pretrained(PROJECT_PATH + "/models/" + model_data['slug'])

  load_time = round((time.time() - start) * 1000)
  print(f"Loaded {model_id} in {load_time}ms")

  return pipe.to(device)


def getPipeline(model_id: str, pipeline_type: str):
  pipeclass = None

  # check if pipeline is supported
  if pipeline_type not in PIPELINES:
    raise Exception(f"Pipeline {pipeline_type} not available")

  # check if model id has a preset pipeline
  if model_id in MODEL_DATA:
    model_data = MODEL_DATA[model_id]
    model_pipelines = model_data.get("pipelines", None)

    # if has only one pipeline, use it
    if len(model_pipelines) == 1:
      pipeline_type = model_pipelines[0]
    
    # if has multiple pipelines, check if the requested pipeline is supported
    elif model_pipelines != None and pipeline_type not in model_pipelines:
      raise Exception(f"Pipeline {pipeline_type} not supported for model {model_id}")
  else:
    raise Exception(f"Model {model_id} not found")

  # get pipeline class
  if pipeline_type == "TXT2IMG":
    pipeclass = StableDiffusionPipeline
  elif pipeline_type == "IMG2IMG":
    pipeclass = StableDiffusionImg2ImgPipeline
  elif pipeline_type == "INPAINT":
    pipeclass = StableDiffusionInpaintPipeline
  elif pipeline_type == "CONTROLNET":
    raise Exception("ControlNet not implemented") # not implemented, throw error
  elif pipeline_type == "PIX2PIX":
    pipeclass = StableDiffusionInstructPix2PixPipeline

  return pipeclass
  


# # This is called once when the server starts
def init():
  # load models
  for model_id in MODEL_DATA:
    model_data = MODEL_DATA[model_id]
    print(f"Loading {model_id}...")
    loaded_models[model_id] = loadModel(model_data)
    print(f"Loaded {model_id}")

  # clear cuda cache
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  pass

# # This is the main inference function that is called by the server
def inference(model_inputs):
  model_id = model_inputs.get("model_id", None)
  pipeline = loaded_models.get("pipeline", DEFAULT_PIPELINE)
  prompt = model_inputs.get("prompt", None)
  negative_prompt = model_inputs.get("negative_prompt", None)
  scheduler_id = model_inputs.get("scheduler_id", DEFAULT_SCHEDULER)
  width = model_inputs.get("width", 512)
  height = model_inputs.get("height", 512)
  guidance_scale = model_inputs.get("guidance_scale", 7.5)
  num_inference_steps = model_inputs.get("steps", 30)
  num_images_per_prompt = model_inputs.get("count", 1)

  # special properties to trigger pipelines
  init_image = model_inputs.get("init_image", None)
  strength = model_inputs.get("strength", 0.85)
  mask_image = model_inputs.get("mask_image", None)
  controlnet_hint = model_inputs.get("control_image", None)

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

  # get pipeline
  pipeclass = getPipeline(model_id, pipeline)

  # get scheduler
  scheduler = getScheduler(model_id, scheduler_id)

  pipe = pipeclass.from_pretrained(
    PROJECT_PATH + "/models/" + model_data['slug'],
    torch_dtype=torch.float16 if model_data["precision"] == "fp16" else None,
    revision="fp16" if model_data["revision"] == "fp16" else None,
    scheduler=scheduler,
    use_auth_token=HF_AUTH_TOKEN,
  )

  # Moving the model to the GPU or CPU
  pipe = pipe.to(device) 

  # remove safety checker
  pipe.safety_checker = lambda images, clip_input: (images, False)

  pipe_data = {
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "width": width,
    "height": height,
    "guidance_scale": guidance_scale,
    "generator": generator,
    "num_inference_steps": num_inference_steps,
    "num_images_per_prompt": num_images_per_prompt,
  }

  # add values for specific pipelines
  if pipeline == "IMG2IMG" or pipeline == "INPAINT":
    pipe_data["init_image"] = init_image
    pipe_data["strength"] = strength
  if pipeline == "INPAINT":
    pipe_data["mask_image"] = mask_image
  if pipeline == "CONTROLNET":
    pipe_data["controlnet_hint"] = controlnet_hint
      
  # execute pipeline
  images = pipe( **pipe_data )
      
  # convert to base64
  images_base64 = []
  for image in images:
    images_base64.append("data:image/png;base64," + encodeBase64Image(image))

  # TEMP save images locally for debugging
  for i, image in enumerate(images):
    image.save(f"./{i}.png")

  # clean up
  del pipe
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  return {
    "images": images_base64
  }
