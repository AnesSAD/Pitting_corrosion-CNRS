# DEEP LEARNING PROJECT - Corrosion product recognition using CNN and computer vision

This project has the objective of applying computer vision techniques and CNN in order to accurately recognize corrosion product seen on microscopic images.
The idea is using computer vision in order to annotate corrosion particles semi-automatically. Then build a CNN from scratch that fits best the prediction of these particles.

## Objectives

- Pre-treatment of microscopic images
- Particle detection using open-cv and semi-automatic annotation (0:no Particle, 1:Particle)
- Build a convolutional neural network from scratch (Unet++)
- Performance evaluation of the model (Dice score, accuracy, precision...)
- Fine-tuning of the model (hyper-parameters, architecture)
- Visualisation of the results and tracking performances of the model

## Structure

📁 Videos/ – data [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.14653184)  
📁 dataset/ – 110 images dataset  
📁 src/ – main code  
📁 notebooks/ – Initial exploration  
📁 models/ – Weights and metrics
📄 README.md  
📄 requirements.txt  

## Dependencies 

- Pytorch
- Numpy
- OpenCV
- Scikit-learn
- Matplotlib

## Dataset file stucture
Raw images and ground thruth masks needs to be in the same folder. Masks should have the mention "mask" in their name and each mask should correspond to a raw image. The dataset file should be structured as follows :  

```python

dataset/  
├── image_1.png  
├── mask_1.png  
├── image_2.png  
├── mask_2.png  
├── image_3.png  
├── mask_3.png
...
```

## How to use it
After corretly structuring you dataset file, execute the training with the following line in you terminal.

```bash
python main.py --dataset_dir "path to your dataset"
```
    
## Author 

Anes SADAOUI -  In the context of a Master 2 (2024/2025) intership at CNRS under the supervision of Slava  SHKIRSKIY.


