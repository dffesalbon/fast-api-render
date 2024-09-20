from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor

import torch
import os
import cv2
import supervision as sv
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
HOME = os.getcwd()
print("HOME:", HOME)
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth") ## sam weights location

def scale_image(image_bytes):
  size_scale = 800
  image_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
  #image_bgr = cv2.imread(image_path)

  h, w = image_bgr.shape[:2]
  aspect_ratio = w / h
  new_height = int(size_scale / aspect_ratio)
  new_width = int(size_scale * aspect_ratio)

  image_bgr = cv2.resize(image_bgr, (new_width, new_height))
  image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

  return image_bgr, image_rgb

def generate_mask(image_rgb, mask_generator):
  sam_result = mask_generator.generate(image_rgb)
  return sam_result


def get_annotated_image(mask_annotator, sam_result, image_bgr):
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    # print("Image shape:", image_bgr.shape)
    # print("Mask shape:", detections.mask.shape)
    return annotated_image



def crop_largest_segmentation(sam_result, annotated_image, image_bgr):
  ### Getting the largest segmented Area

  # Sort the masks by area in descending order
  sorted_masks = sorted(sam_result, key=lambda x: x['area'], reverse=True)
  # Get the largest segmentation area
  largest_mask = sorted_masks[0]
  largest_segmentation = largest_mask['segmentation']

  # Create a blank image of the same size as the original
  largest_mask_image = np.zeros_like(image_bgr)
  # Set the pixels corresponding to the largest segmentation to white
  largest_mask_image[largest_segmentation] = (255, 255, 255)
  # Overlay the mask on the annotated image
  result_image = cv2.addWeighted(annotated_image, 0.7, largest_mask_image, 0.3, 0)

  ### Cropping the largest segmented Area

  # Find contours in the largest segmentation mask
  contours, _ = cv2.findContours(largest_segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Get the bounding rectangle of the largest contour and crop the original image
  largest_contour = max(contours, key=cv2.contourArea)
  x, y, w, h = cv2.boundingRect(largest_contour)
  cropped_image = image_bgr[y:y+h, x:x+w]

  ### Removing the background
  cropped_image = np.zeros_like(image_bgr)
  cropped_image[largest_segmentation] = image_bgr[largest_segmentation]

  return cropped_image

def process_image(image_bytes, mask_generator, mask_annotator):
  # read/get the file image
  image_bgr, image_rgb = scale_image(image_bytes)

  # generate mask
  sam_result = generate_mask(image_rgb, mask_generator)

  # get the annotated image
  annotated_image = get_annotated_image(mask_annotator, sam_result, image_bgr)

  # crop the largest segmentation
  cropped_image = crop_largest_segmentation(sam_result, annotated_image, image_bgr)

  return cropped_image


def segment_image(image_bytes):
  sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
  mask_generator = SamAutomaticMaskGenerator(sam)
  mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
  segmented_image = process_image(image_bytes, mask_generator, mask_annotator)
  print('segment_image')