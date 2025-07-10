import torch
import argparse
from config import*
from dataset import*
from model import*
from train import*
from vizualization import*
from preprocessing import*
from metrics import*
from torch import utils


def main(dataset_dir):
    
    # Importing images and masks
    path = dataset_dir #path to dataset folder
    all_images = sorted([os.path.join(path,img) for img in sorted(os.listdir(path)) if img.endswith('.png') and 'mask' not in img])
    all_masks = sorted([os.path.join(path,img) for img in sorted(os.listdir(path)) if img.endswith('.png') and 'mask' in img])

    # Model parameters 
    DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = cfg['batch_size']
    LR = cfg['lr']
    EPOCHS = cfg['epochs']
    ENCODER = cfg['encoder']
    ENCODER_WEIGHTS = cfg['weights']
    RESIZE = cfg['resize']
    
    # Train set and test set
    trainset = ParticulSegmentation(all_images[:70],all_masks[:70],get_train_augs(RESIZE))
    testset = ParticulSegmentation(all_images[70:],all_masks[70:],get_train_augs(RESIZE))
    
    # Train and test data loader
    dloader_train = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
    dloader_test = torch.utils.data.DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False)
    
    iterator = iter(dloader_train)
    images, masks, image_paths, mask_paths = next(iterator)
    
    # Model 
    model, loss, optimizer = create_model(cfg)
    
    training_loss, testing_loss = training(cfg,os.path.join('..','model'), "weights", dloader_train, dloader_test, model, optimizer, loss )
    
    
    #Loss function
    plot_metrics(os.path.join('..', 'model'), "Loss function", training_loss, testing_loss, cfg)
    
    # Predictions
    show_predictions(dloader_test, model, nbr_images=3, os.path.join('..','model'))
    
    #Compute metrics
    compute_metrics(dloader_test, model, os.path.join('..','model'))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args.dataset_dir)
    

    





