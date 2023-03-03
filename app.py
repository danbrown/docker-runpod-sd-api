from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline, ControlNetModel
import torch
import base64
from io import BytesIO
import PIL
import os
import time
import gc
import requests
from huggingface_hub.repocard import RepoCard

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
SAVE_IMAGES = os.getenv("SAVE_IMAGES", True)

# models
MODELS_DATA = {
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
  "timbrooks/instruct-pix2pix":{
    "model_id": "timbrooks/instruct-pix2pix",
    "slug": "instruct-pix2pix",
    "precision": "fp16",
    "revision": "fp16",
    "pipelines": ["PIX2PIX"]
  }
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

# ControlNet Pipelines
CONTROLNET_MODELS = {
  "CANNY": {
    "model_id": "lllyasviel/sd-controlnet-canny",
    "slug": "controlnet-canny",
    "precision": None,
    "revision": None,
  },
  "DEPTH": {
    "model_id": "lllyasviel/sd-controlnet-depth",
    "slug": "controlnet-depth",
    "precision": None,
    "revision": None,
  }
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

# vars
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_models = {}
loaded_controlnet_models = {}

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

def loadControlNetModel(model_data: str):
  model_id = model_data["model_id"]

  model_folder_exists = os.path.exists(PROJECT_PATH + "/models/" + model_data['slug'])

  # load the model, if is already saved, load from folder, otherwise load from huggingface
  print(f"Loading ControlNet {model_id} from {'pre_saved' if model_folder_exists else 'huggingface'}...")
  model = ControlNetModel.from_pretrained(
    model_id if not model_folder_exists else (PROJECT_PATH + "/models/" + model_data['slug']),
    torch_dtype=torch.float16 if model_data["precision"] == "fp16" else None,
    revision="fp16" if model_data["revision"] == "fp16" else None,
    use_auth_token=HF_AUTH_TOKEN,
  )

  # save model if it is not already saved
  if not model_folder_exists:
    model.save_pretrained(PROJECT_PATH + "/models/" + model_data['slug'])
  
  return model

# this will get the correct pipeline for the model
def getPipeline(model_id: str, pipeline_type: str, controlnet_model_id: str = None):
  pipeclass = None

  # check if pipeline is supported
  if pipeline_type not in PIPELINES:
    raise Exception(f"Pipeline {pipeline_type} not available")

  # check if model id has a preset pipeline
  if model_id in MODELS_DATA:
    model_data = MODELS_DATA[model_id]
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
    pipeclass = StableDiffusionControlNetPipeline
  elif pipeline_type == "PIX2PIX":
    pipeclass = StableDiffusionInstructPix2PixPipeline

  return pipeclass

def getControlnet(model_id: str):
  if model_id in CONTROLNET_MODELS:
    model_data = CONTROLNET_MODELS[model_id]
    
    if controlnet_model_id == None:
      raise Exception(f"ControlNet not found for model {model_id}")
    
    return loadControlNetModel(model_data)
  else:
    raise Exception(f"CotrolNet not found for model {model_id}")

# it will download the image ir is a url, or decode it if it is a base64 string. Then it will resize it to match the required inputs
def normalizeImage(image, width, height) -> PIL.Image:
  if image != None:
    # image is a url
    if 'http' in image: 
        response = requests.get(image)
        image = PIL.Image.open(BytesIO(response.content))
    # image is a base64 data
    elif 'data:image' in image:
        image = PIL.Image.open(BytesIO(base64.b64decode(image).split(',')[1]))
    # image is a base64 string splited
    else:
        image = decodeBase64Image(image)
    # resize image to match required inputs
    image = image.resize((width, height), resample=PIL.Image.LANCZOS)

  return image

# clear cuda cache
def clearCache():
  if torch.cuda.is_available():
    with torch.no_grad():
      torch.cuda.empty_cache()
  gc.collect()
  

# # This is called once when the server starts
def init():

  # load diffusion models
  for model_id in MODELS_DATA:
    model_data = MODELS_DATA[model_id]
    print(f"Loading {model_id}...")
    loaded_models[model_id] = loadModel(model_data)
    print(f"Loaded {model_id}")
  
  # load controlnet models
  for model_id in CONTROLNET_MODELS_DATA:
    model_data = CONTROLNET_MODELS_DATA[model_id]
    print(f"Loading ControlNet {model_id}...")
    loaded_controlnet_models[model_id] = loadControlNetModel(model_data)
    print(f"Loaded ControlNet {model_id}")

  # clear cuda cache
  clearCache()

  pass

# # This is the main inference function that is called by the server
def inference(model_inputs):
  model_id = model_inputs.get("model_id", None)
  pipeline = model_inputs.get("pipeline", DEFAULT_PIPELINE)
  controlnet_type = model_inputs.get("controlnet_type", None)
  prompt = model_inputs.get("prompt", None)
  negative_prompt = model_inputs.get("negative_prompt", None)
  lora_model = model_inputs.get("lora_model", None)
  scheduler_id = model_inputs.get("scheduler_id", DEFAULT_SCHEDULER)
  width = model_inputs.get("width", 512)
  height = model_inputs.get("height", 512)
  guidance_scale = model_inputs.get("guidance_scale", 7.5)
  num_inference_steps = model_inputs.get("steps", 30)
  num_images_per_prompt = model_inputs.get("count", 1)
  safe_mode = model_inputs.get("safe_mode", True)

  # special properties to trigger pipelines
  init_image = normalizeImage(model_inputs.get("init_image", None), width, height)
  mask_image = normalizeImage(model_inputs.get("mask_image", None), width, height)
  strength = model_inputs.get("strength", 0.85)
  image_guidance = model_inputs.get("image_guidance", 1.5)

  # seed generator
  seed = model_inputs.get("seed", None)
  if seed == None:
      generator = torch.Generator(device=device)
      generator.seed()
      seed = generator.initial_seed()
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
  if model_id not in MODELS_DATA.keys():
    return {
      "error": {
        "code": "INVALID_MODEL",
        "message": "Invalid model"
      }
    }

  # if model is not loaded, load it
  if model_id not in loaded_models:
    loaded_models[model_id] = loadModel(model_id)

  # get model data
  model_data = MODELS_DATA[model_id]

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
  if not safe_mode:
    pipe.safety_checker = None

  # general pipe input data
  pipe_data = {
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "guidance_scale": guidance_scale,
    "generator": generator,
    "num_inference_steps": num_inference_steps,
    "num_images_per_prompt": num_images_per_prompt,
  }

  # add values for specific pipelines, to avoid errors
  if pipeline == "TXT2IMG":
    pipe_data["width"] = width
    pipe_data["height"] = height
  
  if pipeline == "IMG2IMG":
    pipe_data["image"] = init_image
    pipe_data["strength"] = strength
  
  if pipeline == "INPAINT":
    pipe_data["image"] = init_image
    pipe_data["mask_image"] = mask_image
    pipe_data["strength"] = strength
  
  if pipeline == "CONTROLNET":
    pipe_data["image"] = init_image
    
    pipeclass.controlnet = getControlnet(model_id)

  if pipeline == "PIX2PIX":
    pipe_data["image"] = init_image
    pipe_data["image_guidance_scale"] = image_guidance

  # load lora model
  # if lora_model != None:
  #   lora_card = RepoCard.load(lora_model)
  #   print(lora_card.data.to_dict())
  #   if lora_card is None or lora_card.data.to_dict()["base_model"] is None:
  #     try:
  #       pipe.unet.load_attn_procs(lora_model)
  #     except:
  #       return {
  #         "error": {
  #           "code": "INVALID_LORA_MODEL",
  #           "message": "Provided lora model is not valid"
  #         }
  #       }
  #   else:
  #     lora_base_model = lora_card.data.to_dict()["base_model"]
  #     if model_id != lora_base_model:
  #       return {
  #         "error": {
  #           "code": "INVALID_LORA_MODEL",
  #           "message": f"Provided lora model is only compatible with {lora_base_model}, but {model_id} was provided"
  #         }
  #       }
  #     else:
  #       pipe.unet.load_attn_procs(lora_model)
      
  # execute pipeline
  images = pipe( **pipe_data ).images
      
  # convert to base64
  images_base64 = []
  for image in images:
    images_base64.append("data:image/png;base64," + encodeBase64Image(image))

  # TEMP save images locally for debugging
  if SAVE_IMAGES:
    for i, image in enumerate(images):
      image.save(f"./{i}.png")

  # clean up
  del pipe
  clearCache()

  return {
    "model_id": model_id,
    "pipeline": pipeline,
    "initial_seed": seed,
    "images": images_base64
  }
