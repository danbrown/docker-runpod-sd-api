import os
import torch

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
SAVE_IMAGES = os.getenv("SAVE_IMAGES", False)
OPTIMIZE = os.getenv("OPTIMIZE", torch.cuda.is_available())

# models
MODELS_DATA = {
  # base model for stable diffusion
  "runwayml/stable-diffusion-v1-5": {
    "model_id": "runwayml/stable-diffusion-v1-5",
    "slug": "sd-v1-5",
    "precision": "fp16",
    "revision": "fp16",
    "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  },

  # Special model for pix2pix
  "timbrooks/instruct-pix2pix":{
    "model_id": "timbrooks/instruct-pix2pix",
    "slug": "instruct-pix2pix",
    "precision": "fp16",
    "revision": "fp16",
    "pipelines": ["PIX2PIX"]
  },

  # Special model for upscale
  "stabilityai/stable-diffusion-x4-upscaler":{
    "model_id": "stabilityai/stable-diffusion-x4-upscaler",
    "slug": "sd-upscaler",
    "precision": "fp16",
    "revision": "fp16",
    "pipelines": ["UPSCALE"]
  },

  # Other models
  # "andite/anything-v4.0": {
  #   "model_id": "andite/anything-v4.0",
  #   "slug": "anything-v4.0",
  #   "precision": None,
  #   "revision": None,
  #   "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  # },
  # "andite/pastel-mix": {
  #   "model_id": "andite/pastel-mix",
  #   "slug": "pastel-mix",
  #   "precision": None,
  #   "revision": None,
  #   "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  # },
  # "hassanblend/HassanBlend1.5.1.2": {
  #   "model_id": "hassanblend/HassanBlend1.5.1.2",
  #   "slug": "hassanblend-v1.5.1.2",
  #   "precision": None,
  #   "revision": None,
  #   "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  # },
  # "XpucT/Deliberate": {
  #   "model_id": "XpucT/Deliberate",
  #   "slug": "deliberate",
  #   "precision": None,
  #   "revision": None,
  #   "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  # },

  # # converted models
  # "danbrown/A-to-Zovya-RPG-v1-5":{
  #   "model_id": "danbrown/A-to-Zovya-RPG-v1-5",
  #   "slug": "a-to-zovya",
  #   "precision": None,
  #   "revision": None,
  #   "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  # },
  # "danbrown/AyoniMix_V2":{
  #   "model_id": "danbrown/AyoniMix_V2",
  #   "slug": "ayonimix-v2",
  #   "precision": None,
  #   "revision": None,
  #   "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  # },
  "danbrown/NeverEnding-Dream": {
    "model_id": "danbrown/NeverEnding-Dream",
    "slug": "neverending-dream",
    "precision": None,
    "revision": None,
    "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  },
  # "danbrown/Cheese-daddys-landscape-mix":{
  #   "model_id": "danbrown/Cheese-daddys-landscape-mix",
  #   "slug": "cheese-daddys-landscape-mix",
  #   "precision": None,
  #   "revision": None,
  #   "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  # },
  # "danbrown/testing-1":{
  #   "model_id": "danbrown/testing-1",
  #   "slug": "testing-1",
  #   "precision": None,
  #   "revision": None,
  #   "pipelines": ["TXT2IMG", "IMG2IMG", "INPAINT", "CONTROLNET"]
  # },
}

# pipelines
PIPELINES = [
    "TXT2IMG",
    "IMG2IMG",
    "INPAINT",
    "CONTROLNET",
    "PIX2PIX",
    "UPSCALE",
    "CONVERT"
]

DEFAULT_PIPELINE = os.getenv("DEFAULT_PIPELINE", PIPELINES[0])


# ControlNet Pipelines
CONTROLNET_MODELS = {
  "CANNY": {
    "model_id": "lllyasviel/sd-controlnet-canny",
    "slug": "controlnet-canny",
  },
  "MLSD":{
    "model_id": "lllyasviel/sd-controlnet-mlsd",
    "slug": "controlnet-mlsd",
  },
  "OPENPOSE": {
    "model_id": "lllyasviel/sd-controlnet-openpose",
    "slug": "controlnet-openpose",
  },
  "SEMANTIC": {
    "model_id": "lllyasviel/sd-controlnet-seg",
    "slug": "controlnet-semantic",
  },
  "DEPTH": {
    "model_id": "lllyasviel/sd-controlnet-depth",
    "slug": "controlnet-depth",
  },
  "NORMAL": { 
    "model_id": "lllyasviel/sd-controlnet-normal",
    "slug": "controlnet-normal",
  },
  "SCRIBBLE": {
    "model_id": "lllyasviel/sd-controlnet-scribble",
    "slug": "controlnet-scribble",
  },
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