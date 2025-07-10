import os
import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt

def show_images(dloader):
  iterator = iter(dloader)
  images,masks,image_paths,mask_paths = next(iterator)

  #images -> torch.Size([8,1,256,256])

  i = np.random.randint(0,images.shape[0])
  images = images[i,:,:,:]
  #images -> torch.Size([1,256,256])

  image = images.numpy().transpose([1,2,0]) #torch.Size([256,256,1])

  masks = masks[i,:,:,:]
  mask = masks.numpy().transpose([1,2,0])

  fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,6))

  axs[0].imshow(image, cmap='gray')
  axs[0].set_title('Image :'+os.path.basename(image_paths[i]))
  axs[0].axis('off')

  axs[1].imshow(mask, cmap='gray')
  axs[1].set_title('Ground truth')
  axs[1].axis('off')
  
  
def show_predictions(dloader, model, nbr_images = 3, output_dir):

  DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')

  images, masks, images_paths, masks_paths = next(iter(dloader))

  model.eval()
  with torch.no_grad():
    output = torch.sigmoid(model(images.to(DEVICE))) # (8,1,256,256)

  # Sigmoid Threshold
  output = ((output > 0.5)*1.0).detach().cpu().numpy()

  for idx in range(nbr_images):
    orig_image = cv.imread(images_paths[idx])
    predicted_mask = cv.resize(output[idx,0],(orig_image.shape[1],orig_image.shape[0]),interpolation= cv.INTER_NEAREST)
    true_mask = cv.resize(masks[idx,0].numpy(),(orig_image.shape[1],orig_image.shape[0]),interpolation= cv.INTER_NEAREST)
    image = cv.resize(images[idx].numpy().transpose((1,2,0)),(orig_image.shape[1],orig_image.shape[0]),interpolation= cv.INTER_NEAREST)

    fig, axs = plt.subplots(1,3,figsize=(25,25))

    #Display
    axs[0].imshow(image,cmap='gray')
    axs[0].set_title(os.path.basename(images_paths[idx]))
    axs[0].axis('off')

    axs[1].imshow(true_mask,cmap='gray')
    axs[1].set_title('GROUND TRUTH')
    axs[1].axis('off')

    axs[2].imshow(predicted_mask,cmap='gray')
    axs[2].set_title('PREDICTION')
    axs[2].axis('off')

    plt.show()
    plt.savefig(output_dir+'predictions')

def plot_metrics(working_dir, experiment_name, training_loss, test_loss, config):

  num_epoch = config['epochs']
  save_dir = os.path.join(working_dir,experiment_name)
  os.makedirs(save_dir, exist_ok= True)


  #Modifying default font size
  plt.rcParams.update({'font.size' : 14})

  fig, ax = plt.subplots(figsize=(10,6))

  ax.plot(range(num_epoch),training_loss, color='red', label='Train loss')
  ax.plot(range(num_epoch),test_loss, color='blue', label='Test loss')
  ax.legend()
  ax.set_title('All loss')
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Loss')

  plt.tight_layout()
  plt.savefig(os.path.join(save_dir,'loss.png'))
  plt.show()
  
