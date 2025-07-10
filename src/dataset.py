import torch
import cv2 as cv
import albumentations as A


def get_train_augs(IMG_SIZE):
    
    """
    get_train_augs(IMG_SIZE: int) -> albumentations.Compose

    Returns a set of data augmentation transforms to be applied during training.

    📌 Purpose:
    - Enhances model robustness and generalization by introducing variability in the training data
    - Performs standard spatial augmentations suitable for segmentation tasks

    🔧 Parameters:
    - IMG_SIZE (int): Target height and width to resize the input images and masks

    🧪 Transformations (applied in sequence):
    - Resize to (IMG_SIZE, IMG_SIZE) using nearest-neighbor interpolation (preserves mask values)
    - Random horizontal flip with 50% probability
    - Random vertical flip with 50% probability
    - Random rotation within ±45° with 50% probability

    🚀 Returns:
    - A composed Albumentations transform (albumentations.Compose)

    ✅ Example:
        transform = get_train_augs(256)
        augmented = transform(image=image, mask=mask)
    """
    return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE,interpolation=cv.INTER_NEAREST),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=45,p=0.5),
                    ])

def get_test_augs(IMG_SIZE):
    return A.Compose([A.Resize(IMG_SIZE,IMG_SIZE,interpolation=cv.INTER_NEAREST),])
  

class ParticulSegmentation(torch.utils.data.Dataset):
    
    """
    ParticulSegmentation(torch.utils.data.Dataset)

    A custom PyTorch Dataset for loading grayscale microscopy images and their corresponding binary masks 
    for particle segmentation tasks.

    📦 Structure:
    - Takes as input two lists of file paths: `all_images` and `all_masks`
    - Each image is paired with its corresponding mask by zipping the lists
    - Applies optional augmentations (e.g., using Albumentations)

    🔧 Parameters:
    - all_images (List[str]): List of file paths to the input images
    - all_masks (List[str]): List of file paths to the segmentation masks
    - augmentations (callable or None): A function or transform that takes an image and mask 
      and returns the augmented versions (e.g., Albumentations transform)

    ⚙️ Data Loading:
    - Loads images and masks using OpenCV in grayscale mode (`cv.imread(..., 0)`)
    - Normalizes pixel values to [0, 1] by dividing by 255.0
    - Applies augmentations if provided
    - Converts images and masks to PyTorch tensors and adds a channel dimension (C=1)

    📈 Returns (on __getitem__):
    - image (Tensor): Tensor of shape (1, H, W) — normalized grayscale image
    - mask (Tensor): Tensor of shape (1, H, W) — normalized binary mask
    - image_path (str): File path to the original image (for tracking/debugging)
    - mask_path (str): File path to the original mask

    ✅ Example:
        dataset = ParticulSegmentation(images_list, masks_list, augmentations=my_aug)
        image, mask, img_path, msk_path = dataset[0]
    """

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
    


