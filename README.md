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

📁 Videos/ – data  Makogon, A., & SHKIRSKIY, S. (2025). Zenodo. https://doi.org/10.5281/zenodo.14653184  
📁 src/ – main code
  📄 config.py
  📄 dataset.py
  📄 metrics.py
  📄 model.py
  📄 preprocessing.py
  📄 train.py
  📄 vizualisation.py
📁 notebooks/ – Initial exploration  
  📄 test.ipynb
  📄 data_generation.ipynb
📄 README.md  
📄 requirements.txt  

## Dependencies 

- Pytorch
- Numpy
- OpenCV
- Scikit-learn
- Matplotlib

## Status

⌛ In progress... 

## Author 

Anes SADAOUI -  In the context of a Master 2 (2024/2025) intership at CNRS under the supervision of Slava  SHKIRSKIY.


