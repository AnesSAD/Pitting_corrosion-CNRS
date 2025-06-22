import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def train_fn(data_loader, model, optimizer, loss):

  model.train()

  total_loss = 0.0
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  for image, mask, image_path, mask_path in tqdm(data_loader):

    image = image.to(DEVICE)
    mask = mask.to(DEVICE)

    optimizer.zero_grad() # Rest les gradient à 0 après chaque batch

    out = model(image)
    l = loss(out,mask)

    l.backward()
    optimizer.step()

    total_loss += l.item()

  return total_loss / len(data_loader)
  

def eval_fn(data_loader, model, optimizer, loss):

  model.eval()
  total_loss = 0.0
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpa')

  with torch.no_grad(): # Cette fois-ci on ne calcule même pas les grades:

    for image, mask, image_path, mask_path in data_loader:

      image = image.to(DEVICE)
      mask = mask.to(DEVICE)

      out = model(image)
      l = loss(out,mask)

      total_loss += l.item()

  return total_loss / len(data_loader)
 
 
def training(config, working_directory, experiment_name, dloader_train, dloader_test, model, optimizer, loss):

  epochs = config['epochs']
  best_test_loss = np.inf
  training_loss = []
  testing_loss = []

  os.makedirs(os.path.join(working_directory,experiment_name), exist_ok = True)
  writer = SummaryWriter(os.path.join(working_directory,experiment_name))

  for i in range(0,epochs):

    train_loss = train_fn(dloader_train,model,optimizer,loss)
    test_loss = eval_fn(dloader_test, model, optimizer,loss)

    writer.add_scalar("Train loss", train_loss, i)

    writer.add_scalar("Testing loss", test_loss, i)

    training_loss.append(train_loss)
    testing_loss.append(test_loss)

    if test_loss < best_test_loss :

      torch.save(model, os.path.join(working_directory,experiment_name,'best_model_microscope.pth'))
      print("SAVE MODEL WITH LOSS :",test_loss)
      best_test_loss = test_loss
      plot_figures_in_training(dloader_test, model)

    print(f"Epoch {i+1} Train Loss {train_loss} Test loss {test_loss}")

  writer.close()

  return training_loss, testing_loss
  

def plot_figures_in_training(loader,model):

  DEVICE = torch.device('cuda' if torch.cuda.is_available() else cpu)

  with torch.no_grad():
    for images, masks, image_path, masks_path in loader:

      image, mask = images.to(DEVICE) , masks.to(DEVICE)
      out = torch.sigmoid(model(image))


      image, mask = process_images(images[0]), process_images(mask[0])
      pred_mask = process_images((out[0] > 0.5)*1 )

      fig, axs = plt.subplots(2,2,figsize=(6,6))
      titles = ['Original image', 'Prédicted mask', 'True Masks','Overlay :  Mask on the original image']

      plot_image(axs[0,0], image, titles[0], cmap='gray')
      plot_image(axs[0,1], pred_mask, titles[1], cmap='gray')
      plot_image(axs[1,0], mask, titles[2], cmap='gray')
      plot_image(axs[1,1], image, titles[3], cmap='gray')
      plot_image(axs[1,1], pred_mask, titles[3], alpha=0.4, cmap='gray')

      plt.tight_layout()
      plt.show()

      break
      
def plot_image(ax, img, title, cmap=None, alpha=None):

  ax.imshow(img,cmap=cmap,alpha=alpha)
  ax.set_title(title)
  ax.axis('off')

def process_images(tensor) :
  array = tensor.detach().cpu().numpy().transpose((1,2,0))
  return array
