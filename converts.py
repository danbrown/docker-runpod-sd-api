import torch
import cv2
import os
from PIL import Image
import numpy as np
import time
from controlnet_aux import MLSDdetector, OpenposeDetector, HEDdetector
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation

# config
from config import PROJECT_PATH

# It takes an image, converts it to grayscale, and then applies the Canny edge detection algorithm to it
def imageToCanny(image, low_threshold=100, high_threshold=200):
  start = time.time()
  image = np.array(image)

  image = cv2.Canny(image, low_threshold, high_threshold)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  image = Image.fromarray(image)
  
  diff = round((time.time() - start) * 1000)
  print(f"convert imageToCanny took {diff}ms")

  return image

# It takes an image and converts it MLSD lines
def imageToMLSDLines(image):
  start = time.time()
  mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

  image = mlsd(image)

  del mlsd

  diff = round((time.time() - start) * 1000)
  print(f"convert imageToMLSDLines took {diff}ms")

  return image

# It takes an image and converts it to Openpose keypoints
def imageToOpenPose(image):
  start = time.time()
  openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

  image = openpose(image)

  del openpose

  diff = round((time.time() - start) * 1000)
  print(f"convert imageToOpenPose took {diff}ms")

  return image

def imageToSemanticSegmentation(image):
  start = time.time()

  image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
  image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

  pixel_values = image_processor(image, return_tensors="pt").pixel_values
  with torch.no_grad():
    outputs = image_segmentor(pixel_values)

  seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

  color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

  # palette = np.array(ade_palette())

  # for label, color in enumerate(palette):
  #   color_seg[seg == label, :] = color

  # color_seg = color_seg.astype(np.uint8)

  del image_processor
  del image_segmentor

  image = Image.fromarray(color_seg)

  diff = round((time.time() - start) * 1000)
  print(f"convert imageToSemanticSegmentation took {diff}ms")

  return image

# It takes an image and converts it to a depth map
def imageToDepthMap(image):
  start = time.time()

  depth_estimator = pipeline('depth-estimation')
  image = depth_estimator(image)['depth']
  image = np.array(image)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  image = Image.fromarray(image)

  del depth_estimator

  diff = round((time.time() - start) * 1000)
  print(f"convert imageToDepthMap took {diff}ms")

  return image

# It takes an image and converts it to a normal map
def imageToNormalMap(image):
  start = time.time()

  depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas" )

  image = depth_estimator(image)['predicted_depth'][0]

  image = image.numpy()

  image_depth = image.copy()
  image_depth -= np.min(image_depth)
  image_depth /= np.max(image_depth)

  bg_threhold = 0.4

  x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
  x[image_depth < bg_threhold] = 0

  y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
  y[image_depth < bg_threhold] = 0

  z = np.ones_like(x) * np.pi * 2.0

  image = np.stack([x, y, z], axis=2)
  image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
  image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
  image = Image.fromarray(image)

  del depth_estimator

  diff = round((time.time() - start) * 1000)
  print(f"convert imageToNormalMap took {diff}ms")

  return image

# It takes an image and converts it to a scribble image
def imageToScribble(image):
  start = time.time()

  hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
  image = hed(image, scribble=True)

  del hed

  diff = round((time.time() - start) * 1000)
  print(f"convert imageToScribble took {diff}ms")

  return image

# It takes an image and converts it to a HED boundary map
def imageToHED(image):
  start = time.time()

  hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
  image = hed(image)

  del hed

  diff = round((time.time() - start) * 1000)
  print(f"convert imageToHED took {diff}ms")

  return image