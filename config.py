import os
import torch

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
SAVE_IMAGES = os.getenv("SAVE_IMAGES", False)
OPTIMIZE = os.getenv("OPTIMIZE", torch.cuda.is_available())

# models
MODELS_DATA = {
  "runwayml/stable-diffusion-v1-5": {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "slug": "sd-v1-5",
    "precision": "fp16",
    "revision": "fp16",
    "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  },
  "andite/anything-v4.0": {
    "model_id": "andite/anything-v4.0",
    "slug": "anything-v4.0",
    "precision": None,
    "revision": None,
    "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  },
  # "timbrooks/instruct-pix2pix":{
  #   "model_id": "timbrooks/instruct-pix2pix",
  #   "slug": "instruct-pix2pix",
  #   "precision": "fp16",
  #   "revision": "fp16",
  #   "pipelines": ["PIX2PIX"]
  # },
  "darkstorm2150/Protogen_x3.4_Official_Release":{
    "model_id": "darkstorm2150/Protogen_x3.4_Official_Release",
    "slug": "protogen-x3.4",
    "precision": None,
    "revision": None,
    "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
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


# ControlNet Pipelines
CONTROLNET_MODELS = {
  "CANNY": {
    "model_id": "lllyasviel/sd-controlnet-canny",
    "slug": "controlnet-canny",
  },
  # "MLSD":{
  #   "model_id": "lllyasviel/sd-controlnet-mlsd",
  #   "slug": "controlnet-mlsd",
  # },
  "OPENPOSE": {
    "model_id": "lllyasviel/sd-controlnet-openpose",
    "slug": "controlnet-openpose",
  },
  # "SEMANTIC": {
  #   "model_id": "lllyasviel/sd-controlnet-seg",
  #   "slug": "controlnet-semantic",
  # },
  "DEPTH": {
    "model_id": "lllyasviel/sd-controlnet-depth",
    "slug": "controlnet-depth",
  },
  "NORMAL": { 
    "model_id": "lllyasviel/sd-controlnet-normal",
    "slug": "controlnet-normal",
  },
  # "SCRIBBLE": {
  #   "model_id": "lllyasviel/sd-controlnet-scribble",
  #   "slug": "controlnet-scribble",
  # },
  "HED": {
    "model_id": "lllyasviel/sd-controlnet-hed",
    "slug": "controlnet-hed",
  },
}

# schedulers
SCHEDULERS = [
    "DPMSolverMultistepScheduler",
    "UniPCMultistepScheduler",
    "LMSDiscreteScheduler",
    "DDIMScheduler",
    "PNDMScheduler",
    "EulerAncestralDiscreteScheduler",
    "EulerDiscreteScheduler",
]

DEFAULT_SCHEDULER = os.getenv("DEFAULT_SCHEDULER", SCHEDULERS[0])