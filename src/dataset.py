import matplotlib.pyplot as plt
import numpy as np
import random
import os
import skimage.io
import cv2 as cv
import torch
import albumentations as A
import seaborn as sns

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss


def get_train_augs(IMG_SIZE):
  return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE,interpolation=cv.INTER_NEAREST),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=45,p=0.5),
                    ])

def get_test_augs(IMG_SIZE):
  return A.Compose([A.Rezise(IMG_SIZE,IMG_SIZE,interpolation=cv.INTER_NEAREST),])
  

class ParticulSegmentation(torch.utils.data.Dataset):

  def __init__(self,all_images,all_masks,augmentations):
    self.data = list(zip(all_images,all_masks))
    self.augmentations = augmentations

  def __len__(self):
    return len(self.data)

  def __getitem__(self,index):
    image_path, mask_path = self.data[index]
    image = cv.imread(image_path,0).astype(float) / 255.0
    mask = cv.imread(mask_path,0).astype(float) / 255.0

    if self.augmentations:
      data = self.augmentations(image=image,mask=mask)
      image,mask= data['image'],data['mask']

    return torch.from_numpy(image).unsqueeze(0).float(),torch.from_numpy(mask).unsqueeze(0).float(),image_path,mask_path
    


