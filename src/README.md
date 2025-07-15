# Source code structure for image annotation, model training and inference


- config.py : parameters for model training (image size, learning rate...)
- dataset.py : class and functions for dataloading and image augmentation
- inference.py : main executable code for single image inference. Contains image slicing, inference and image reconstruction functions
- main.py : main executable code for model training. Contains model initialization, forward, backward propagation function, evaluation and saving of performances metrics. Saving of weights.
- metrics.py : contains function to compute F1 score, accuracy ... metrics.
- model.py : contain U-Net model architecture.
- train.py : functions for model training and evaluation
- visualization.py : Display functions. Displaying images and predictions, predictions while training. Displaying loss function...
