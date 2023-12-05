import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from review import Review

import random
random.seed(33)
torch.manual_seed(0)
reviews = []

# Считываем данные из файлов
with open('dataset.csv', 'r', encoding='utf-8') as f:
    dataset = f.read().split('\n')

for item in dataset[1:4 + 1]:
    #print(item)
    reviews.append(Review(item))

for review in reviews:
    print(review.text)
