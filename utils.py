import torch
import base64
from io import BytesIO
import PIL
import os
import time
import gc
import requests
import numpy as np
import sys
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decodeBase64Image(imageStr: str, name: str) -> PIL.Image:
  image = PIL.Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))
  print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
  return image

def encodeBase64Image(image: PIL.Image) -> str:
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  return base64.b64encode(buffered.getvalue()).decode("utf-8")

# it will download the image ir is a url, or decode it if it is a base64 string. Then it will resize it to match the required inputs
def normalizeImage(image, width, height) -> PIL.Image:
  if image != None:
    # image is a url
    if 'http' in image: 
        response = requests.get(image)
        image = PIL.Image.open(BytesIO(response.content))
    # image is a base64 data
    elif 'data:image' in image:
        image = decodeBase64Image(image.split(',')[1], 'base64')
    # image is a base64 string splited
    else:
        image = decodeBase64Image(image, 'base64')
    # resize image to match required inputs
    image = image.resize((width, height), resample=PIL.Image.Resampling.LANCZOS)

    return image
  else:
    return None

# clear cuda cache
def clearCache():
  if torch.cuda.is_available():
    with torch.no_grad():
      torch.cuda.empty_cache()
  gc.collect()

def clearHuggingFaceCache():
  cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
  if os.path.exists(cache_dir):
    sys.stdout.write(f"Deleting cache: {cache_dir}")
    shutil.rmtree(cache_dir)
