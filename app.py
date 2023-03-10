from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import StableDiffusionUpscalePipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import schedulers as _schedulers
import torch
import base64
from io import BytesIO
import PIL
import os
import time
import gc
import requests
from huggingface_hub.repocard import RepoCard

# converts
from converts import imageToCanny, imageToMLSDLines, imageToOpenPose, imageToSemanticSegmentation, imageToDepthMap, imageToNormalMap, imageToScribble, imageToHED

# utils
from utils import device, decodeBase64Image, encodeBase64Image, normalizeImage, clearCache, clearHuggingFaceCache

# config
from config import PROJECT_PATH, HF_AUTH_TOKEN, OPTIMIZE, SAVE_IMAGES, MODELS_DATA, PIPELINES, DEFAULT_PIPELINE, CONTROLNET_MODELS, SCHEDULERS, DEFAULT_SCHEDULER


# load model and save it locally
def loadModel(model_data: str, download: bool = False):
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

  # run optimizations or move to device if disabled
  if OPTIMIZE:
    # optimize pipe using cpu offloading
    pipe.enable_model_cpu_offload()

    # enable xformers memory efficient
    pipe.enable_xformers_memory_efficient_attention()
  else:
    # Moving the model to the GPU or CPU
    pipe = pipe.to(device) 
  
  # save model if it is not already saved
  if not model_folder_exists and download:
    pipe.save_pretrained(PROJECT_PATH + "/models/" + model_data['slug'])


  # remove huggingface cache
  clearHuggingFaceCache()

  load_time = round((time.time() - start) * 1000)
  print(f"Loaded {model_id} in {load_time}ms")

  return pipe

def loadControlNetModel(controlnet_data: str, model_data:str = None, download: bool = False):
  controlnet_id = controlnet_data["model_id"]

  model_folder_exists = os.path.exists(PROJECT_PATH + "/models/" + controlnet_data['slug'])

  # load the model, if is already saved, load from folder, otherwise load from huggingface
  print(f"Loading ControlNet {controlnet_id} from {'pre_saved' if model_folder_exists else 'huggingface'}...")
  controlnet_model = ControlNetModel.from_pretrained(
    controlnet_id if not model_folder_exists else (PROJECT_PATH + "/models/" + controlnet_data['slug']),
    torch_dtype=torch.float16 if model_data and model_data["precision"] == "fp16" else None,
    revision="fp16" if model_data and model_data["revision"] == "fp16" else None,
    use_auth_token=HF_AUTH_TOKEN,
  )

  # save model if it is not already saved
  if not model_folder_exists and download:
    controlnet_model.save_pretrained(PROJECT_PATH + "/models/" + controlnet_data['slug'])

  # remove huggingface cache
  clearHuggingFaceCache() 
  
  return controlnet_model

# this will get the correct pipeline for the model
def getPipeline(model_id: str, pipeline_type: str):
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
  elif pipeline_type == "UPSCALE":
    pipeclass = StableDiffusionUpscalePipeline

  return pipeclass

def getControlnet(controlnet_type: str, model_data: str):
  if controlnet_type in CONTROLNET_MODELS:
    controlnet_data = CONTROLNET_MODELS[controlnet_type]
    
    if controlnet_data == None:
      raise Exception(f"ControlNet not found for model {controlnet_type}")
    
    return loadControlNetModel(controlnet_data, model_data)
  else:
    raise Exception(f"ControlNet not found for model {controlnet_type}")

def getScheduler(scheduler_id: str, config) -> str:
  print(f"Initializing {scheduler_id}...")

  start = time.time()

  scheduler = getattr(_schedulers, scheduler_id)
  if scheduler == None:
    raise Exception(f"Scheduler {scheduler_id} not found")

  inittedScheduler = scheduler.from_config(
    config,
    use_auth_token=HF_AUTH_TOKEN,
  )

  diff = round((time.time() - start) * 1000)
  print(f"Initialized {scheduler_id} in {diff}ms")

  return inittedScheduler


# # This is called once when the server starts
def init():

  # create models folder if it does not exist
  if not os.path.exists(PROJECT_PATH + "/models"):
    os.mkdir(PROJECT_PATH + "/models")

  # load diffusion models
  for model_id in MODELS_DATA:
    model_data = MODELS_DATA[model_id]
    print(f"Loading {model_id}...")
    model = loadModel(model_data, download=True)
    del model
    print(f"Loaded {model_id}")

  # clear cuda cache
  clearCache()
  
  # load controlnet models
  for model_id in CONTROLNET_MODELS:
    model_data = CONTROLNET_MODELS[model_id]
    print(f"Loading ControlNet {model_id}...")
    model = loadControlNetModel(model_data, download=True)
    del model
    print(f"Loaded ControlNet {model_id}")

  # clear cuda cache
  clearCache()

  pass

# # This is the main inference function that is called by the server
def inference(model_inputs):
  pipeline = model_inputs.get("pipeline", DEFAULT_PIPELINE)

  # if pipeline is "CONVERT", run the convert function instead
  if pipeline == "CONVERT":
    return convert(model_inputs)

  model_id = model_inputs.get("modelId", None)
  controlnet_type = model_inputs.get("controlnetType", None)
  prompt = model_inputs.get("prompt", None)
  negative_prompt = model_inputs.get("negativePrompt", None)
  lora_model = model_inputs.get("loraModel", None)
  scheduler_id = model_inputs.get("schedulerId", DEFAULT_SCHEDULER)
  width = model_inputs.get("width", 512)
  height = model_inputs.get("height", 512)
  guidance_scale = model_inputs.get("guidanceScale", 7.5)
  num_inference_steps = model_inputs.get("steps", 30)
  num_images_per_prompt = model_inputs.get("count", 1)
  safe_mode = model_inputs.get("safeMode", True)

  # special properties to trigger pipelines
  init_image = normalizeImage(model_inputs.get("initImage", None), width, height)
  mask_image = normalizeImage(model_inputs.get("maskImage", None), width, height)
  strength = model_inputs.get("strength", 0.85)
  image_guidance = model_inputs.get("imageGuidance", 1.5)

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


  # get model data
  model_data = MODELS_DATA[model_id]

  # get pipeline class based on pipeline type
  pipeclass = getPipeline(model_id, pipeline)

  # init the pipeline
  pipe = pipeclass.from_pretrained(
    PROJECT_PATH + "/models/" + model_data['slug'],
    torch_dtype=torch.float16 if model_data["precision"] == "fp16" else None,
    revision="fp16" if model_data["revision"] == "fp16" else None,
    use_auth_token=HF_AUTH_TOKEN,
    controlnet=getControlnet(controlnet_type, model_data) if pipeline == "CONTROLNET" else None,
  )

  # get scheduler
  pipe.scheduler = getScheduler(scheduler_id, pipe.scheduler.config)

  # run optimizations or move to device if disabled
  if OPTIMIZE:
    # optimize pipe using cpu offloading
    pipe.enable_model_cpu_offload()

    # enable xformers memory efficient
    pipe.enable_xformers_memory_efficient_attention()
  else:
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
    "modelId": model_id,
    "pipeline": pipeline,
    "initialSeed": seed,
    "images": images_base64
  }

# # This is the convert function that is called by the server
def convert(model_inputs):

  width = model_inputs.get("width", 512)
  height = model_inputs.get("height", 512)

  init_image = normalizeImage(model_inputs.get("initImage", None), width, height)
  controlnet_type = model_inputs.get("controlnetType", None)

  # check if controlnet is valid, if not return error
  if controlnet_type not in CONTROLNET_MODELS:
    return {
      "error": {
        "code": "INVALID_CONTROLNET_TYPE",
        "message": "Invalid controlnet_type"
      }
    }

  # check if init_image is valid, if not return error
  if init_image == None:
    return {
      "error": {
        "code": "NO_INIT_IMAGE",
        "message": "No init_image provided"
      }
    }

  images_base64 = []

  # covert image to the desired controlnet input
  if controlnet_type == "CANNY":
    init_image = imageToCanny(init_image)
    resized = init_image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS) # resize to fit desired sizes
    images_base64.append("data:image/png;base64," + encodeBase64Image(resized))

  elif controlnet_type == "MLSD":
    init_image = imageToMLSDLines(init_image)
    resized = init_image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS) # resize to fit desired sizes
    images_base64.append("data:image/png;base64," + encodeBase64Image(resized))
    
  elif controlnet_type == "OPENPOSE":
    init_image = imageToOpenPose(init_image)
    resized = init_image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS) # resize to fit desired sizes
    images_base64.append("data:image/png;base64," + encodeBase64Image(resized))

  elif controlnet_type == "SEMANTIC":
    init_image = imageToSemanticSegmentation(init_image)
    resized = init_image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS) # resize to fit desired sizes
    images_base64.append("data:image/png;base64," + encodeBase64Image(resized))

  elif controlnet_type == "DEPTH":
    init_image = imageToDepthMap(init_image)
    resized = init_image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS) # resize to fit desired sizes
    images_base64.append("data:image/png;base64," + encodeBase64Image(resized))

  elif controlnet_type == "NORMAL":
    init_image = imageToNormalMap(init_image)
    resized = init_image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS) # resize to fit desired sizes
    images_base64.append("data:image/png;base64," + encodeBase64Image(resized))

  elif controlnet_type == "SCRIBBLE":
    init_image = imageToScribble(init_image)
    resized = init_image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS) # resize to fit desired sizes
    images_base64.append("data:image/png;base64," + encodeBase64Image(resized))

  elif controlnet_type == "HED":
    init_image = imageToHED(init_image)
    resized = init_image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS) # resize to fit desired sizes
    images_base64.append("data:image/png;base64," + encodeBase64Image(resized))

  # clean up
  del init_image
  clearCache()

  return {
    "controlnetType": controlnet_type,
    "width": width,
    "height": height,
    "images": images_base64
  }

