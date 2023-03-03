import torch
import cv2
import PIL
import numpy as np
from controlnet_utils import ade_palette
from controlnet_aux import MLSDdetector, OpenposeDetector, HEDdetector
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation

# It takes an image, converts it to grayscale, and then applies the Canny edge detection algorithm to it
def imageToCanny(image, low_threshold=100, high_threshold=200):
  
  image = np.array(image)

  image = cv2.Canny(image, low_threshold, high_threshold)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  image = Image.fromarray(image)

  return image

# It takes an image and converts it MLSD lines
def imageToMLSDLines(image):
  mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

  image = mlsd(image)
  return image

# It takes an image and converts it to Openpose keypoints
def imageToOpenpose(image):
  openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

  image = openpose(image)
  return image

def imageToSemanticSegmentation(image):
  image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
  image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

  pixel_values = image_processor(image, return_tensors="pt").pixel_values
  with torch.no_grad():
    outputs = image_segmentor(pixel_values)

  seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

  color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

  palette = np.array(ade_palette())

  for label, color in enumerate(palette):
    color_seg[seg == label, :] = color

  color_seg = color_seg.astype(np.uint8)

  image = Image.fromarray(color_seg)
  return image

# It takes an image and converts it to a depth map
def imageToDepthMap(image):
  depth_estimator = pipeline('depth-estimation')
  image = depth_estimator(image)['depth']
  image = np.array(image)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  image = Image.fromarray(image)
  return image

# It takes an image and converts it to a normal map
def imageToNormalMap(image):
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
  return image

# It takes an image and converts it to a scribble image
def imageToScribble(image):
  hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
  image = hed(image, scribble=True)
  return image

# It takes an image and converts it to a HED boundary map
def imageToHED(image):
  hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
  image = hed(image)
  return image