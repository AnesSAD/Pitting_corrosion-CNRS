import os
import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt

def infer_single_image(image, model, device='cuda'):
    """
    Infer a given image based on weights generated with U-net++
    
    🔧 Parameters:
    - image (numpy array) : considered image on which we want to detect our objects
    - model (pytorch objects)  : trained weights coming from a .pth file
    - device : cuda for gpu (could also use 'cpu')

    🚀 Returns:
    - predicted_mask (list) : predicted mask
        
    """
    
    DEVICE = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    model.eval()

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 1. Reading and pre processing of image
    h, w = image.shape

    # Normalisation [0,1]
    image_input = image.astype(np.float32) / 255.0
    image_input = torch.from_numpy(image_input).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 2. Inference
    with torch.no_grad():
        output = torch.sigmoid(model(image_input))

    # 3. Post-traitement
    output = (output > 0.5).float().cpu().numpy()[0, 0]
    predicted_mask = cv.resize(output, (w, h), interpolation=cv.INTER_NEAREST)

    # # 4. Display
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[0].imshow(image, cmap='gray')
    # axs[0].set_title('Image originale')
    # axs[0].axis('off')

    # axs[1].imshow(predicted_mask, cmap='gray')
    # axs[1].set_title('Masque prédit')
    # axs[1].axis('off')

    # plt.tight_layout()
    # plt.show()

    return predicted_mask

def slice_image(image, patch_size=224, stride=100):
    """
    Slices images of higher resolution into small slices of 224x224 which are fitted to our model which was trained with 224x224 images for imbalance considerations.
    
    🔧 Parameters:
    - image (np.array): considered image
    - patch_size (int, default : 224): considered slices size
    - stride (int, default=100) : corresponds to the number of pixels shifted (horizontaly and verticaly) between 2 images 

    🚀 Returns:
    - patches (list) : list of numpy array corresponding to each slice
        
    """
    h, w = image.shape
    patch_id = 0

    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)

            patch_uint8 = (patch * 255).astype(np.uint8)
            patch_id += 1

    

    return patches

def reconstruct_image(initial_image, slices, stride):
    """
    Reconstruct image of full resolution from multiple slices of the initial images at lower resolution

    🔧 Parameters:
    - initial_image (numpy array) : initial image before slicing 
    - slices (list) : list of numpy arrays corresponding to segmentation masks
    - stride (int) : initial stride used for the slicing of the image


    🚀 Returns:
    - thresh : single numpy array corresponding to the reconstructed segmentation mask
        
    """
    initial_image = cv.cvtColor(initial_image, cv.COLOR_BGR2GRAY)
    h,w = initial_image.shape

    img = np.zeros((h,w))

    patch_size = slices[0].shape[0]

    idx = 0
    for  y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img[y:y + patch_size, x:x + patch_size] += slices[idx]
            idx += 1
    
    _, thresh = cv.threshold(img,0,1,cv.THRESH_BINARY) # We apply a threshold because of the superposition of an object that could be on two different slices therefore its pixel value would be 2 and not 1.
        

    return thresh
    

def main(image_path):
    """Main loop that slices the image, make the inference ont each slice and reconstruct the full image.
    
    🔧 Parameters:
    - image_path = location of the full resolution image
    - 


    🚀 Returns:
    - thresh : single numpy array corresponding to the reconstructed segmentation mask
    """

    train_dir = sorted(os.listdir("../models"))[-1]
    weights_path = "../models/" + train_dir + "/weights/best_model_microscope.pth"
    model = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    image = cv.imread("image_path")
    
    patches = slice_image(image) # Slices are made with a stride of 100 pixels meaning two consecutive image corresponds to a 100 pixels shift
    
    seg_masks = []
    for img in patches:
        mask = infer_single_image(img,model)
        seg_masks.append(mask)
    
    reconstructed_image = reconstruct_image(image,seg_masks,100)


    return reconstructed_image



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    
    args = parser.parse_args()
    
    main(image_path)
    



