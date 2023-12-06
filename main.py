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
texts = []
labels = []


def preprocess(text):
    """"
    Функция чтобы очистить текст отзыва от пунктуации и выделить все слова
    """
    clear_text = "".join([s for s in text if s not in punctuation])  # убираем пунктуацию
    text_words = clear_text.split()  # получаем массив слов
    return clear_text, text_words


# Считываем данные из файлов
with open('dataset.csv', 'r', encoding='utf-8') as f:
    dataset = f.read().split('\n')

for item in dataset[1:4-1]:
    reviews.append(Review(item))

for review in reviews:
    print(preprocess(review.text))
    texts.append(review.text)
    labels.append(review.label)


#all_reviews, all_words = preprocess(texts)
#print('Общее число отзывов: ', len(all_reviews))