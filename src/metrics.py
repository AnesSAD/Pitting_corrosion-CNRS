import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def compute_metrics(dloader, model, output_dir):

  print('COMPUTING METRICS ...')

  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  f1_score_ = []
  accuracy_ = []
  precision_ = []
  recall_ = []


  for image, target, image_path, mask_path in tqdm(dloader):

    image = image.to(DEVICE)

    out = model(image)
    out = torch.sigmoid(out)
    out = (out > 0.5)*1.0
    out = out.cpu().numpy()

    f1 = f1_score(target.cpu().numpy().ravel(), out.ravel(), average='binary', zero_division=0)
    accuracy = accuracy_score(target.cpu().numpy().ravel(), out.ravel())
    recall = recall_score(target.cpu().numpy().ravel(), out.ravel(), average='binary', zero_division=0)
    precision = precision_score(target.cpu().numpy().ravel(), out.ravel(), average='binary', zero_division=0)

    f1_score_.append(f1)
    accuracy_.append(accuracy)
    recall_.append(recall)
    precision_.append(precision)

  score = {'F1_SCORE' : np.mean(f1_score_),
             'ACCURACY' : np.mean(accuracy_),
             'RECALL': np.mean(recall_),
             'PRECISION' : np.mean(precision_)}

  labels = score.keys()
  values = score.values()

  sns.set()
  plt.figure(figsize=(10,6))
  ax = sns.barplot(x=list(labels),y=list(values))
  plt.title('Metrics Results', fontweight='bold')
  plt.xlabel('Metrics', fontweight='bold')
  plt.ylabel('Values', fontweight='bold')

  for i,v in enumerate(values):
    ax.text(i,v+0.002,str(round(v,2)),ha='center',va='bottom')

  plt.show()
  plt.savefig(output_dir + 'metrics')
