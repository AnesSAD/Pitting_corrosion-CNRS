ropimport numpy as np
import cv2 as cv
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from collections import defaultdict

def resize_to_window(frame):
    frame = cv.resize(frame,(frame.shape[0]//3,frame.shape[1]//3))
    return frame

def preprocess_for_model(image):
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #enhanced = clahe.apply(image)
    normalized = enhanced.astype(np.float32) / 255.0
    return normalized
    
def compute_entropy(patch):
    uint_patch = img_as_ubyte(patch)
    return np.mean(entropy(uint_patch, disk(5)))
    
def extract_patches(    ):
    h, w = image.shape[:2]
    patch_id = 0

    patches = defaultdict(list)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches[f"{concentration}_{frame}_{patch_id:.3f}"].append(patch)

            # Entropic filtering
            if compute_entropy(patch) >= entropy_thresh:
                patch_uint8 = (patch * 255).astype(np.uint8)
                patch_id += 1    
    
    return patches
    

