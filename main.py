from multiprocessing import freeze_support
import pickle
import numpy as np
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from review import Review

import random

from utils import preprocess

random.seed(33)
torch.manual_seed(0)
reviews = []
all_texts = []
all_words = []
labels = []


def pad_text(encoded_texts, seq_length):
    padded = []
    for text in encoded_texts:
        if len(text) >= seq_length:
            padded.append(text[:seq_length])
        else:
            padded.append([0] * (seq_length - len(text)) + text)

    return np.array(padded)


class SentimentRNN(nn.Module):
    """
    Соберем модель для классификации текстов
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """
        Инициализируем модель, обозначая слои и гиперпараметры
        """
        super(SentimentRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, h):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, h)
        # print(lstm_out.shape)
        lstm_out = lstm_out[:, -1, :]  # getting the last time step output

        # fully-connected layer
        out = self.fc(lstm_out)
        # sigmoid function
        out = self.sig(out)
        # return last sigmoid output
        return out


class ValueMeter(object):
    """
    Вспомогательный класс, чтобы отслеживать loss и метрику
    """
    def __init__(self):
        self.sum = 0
        self.total = 0

    def add(self, value, n):
        self.sum += value*n
        self.total += n

    def value(self):
        return self.sum/self.total

def log(mode, epoch, loss_meter, accuracy_meter, best_perf=None):
    """
    Вспомогательная функция
    """
    print(
        f"[{mode}] Epoch: {epoch:0.2f}. "
        f"Loss: {loss_meter.value():.2f}. "
        f"Accuracy: {100*accuracy_meter.value():.2f}% ", end="\n")

    if best_perf:
        print(f"[best: {best_perf:0.2f}]%", end="")

def accuracy(outputs, labels):
    preds = torch.round(outputs.squeeze())
    # print(preds, labels)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def trainval(model, loaders, optimizer, epochs=5):
    """
    model: модель, которую собираемся обучать
    loaders: dict с dataloader'ами для обучения и валидации
    optimizer: оптимизатор
    epochs: число обучающих эпох (сколько раз пройдемся по всему датасету)
    """
    loss_meter = {'training': ValueMeter(), 'validation': ValueMeter()}
    accuracy_meter = {'training': ValueMeter(), 'validation': ValueMeter()}

    loss_track = {'training': [], 'validation': []}
    accuracy_track = {'training': [], 'validation': []}

    for epoch in range(epochs):  # итерации по эпохам
        for mode in ['training', 'validation']:  # обучение - валидация
            # считаем градиаент только при обучении:
            with torch.set_grad_enabled(mode == 'training'):
                # в зависимоти от фазы переводим модель в нужный ружим:
                model.train() if mode == 'training' else model.eval()
                for texts, labels in tqdm(loaders[mode]):
                    texts = texts.to(device)  # отправляем тензор на GPU
                    labels = labels.to(device)
                    bs = labels.shape[0]  # размер батча (отличается для последнего батча в лоадере)

                    zero_init = torch.zeros(num_layers, bs, hidden_dim).to(device)
                    h = tuple([zero_init, zero_init])

                    preds = model(texts, h)  # forward pass - прогоняем тензор с картинками через модель

                    loss = nn.BCELoss()(preds.squeeze(), labels.float())

                    loss_meter[mode].add(loss.item(), bs)

                    # если мы в фазе обучения
                    if mode == 'training':
                        optimizer.zero_grad()  # обнуляем прошлый градиент
                        loss.backward()  # делаем backward pass (считаем градиент)
                        optimizer.step()  # обновляем веса

                    acc = accuracy(preds, labels)  # считаем метрику
                    # храним loss и accuracy для батча
                    accuracy_meter[mode].add(acc, bs)

            # в конце фазы выводим значения loss и accuracy
            log(mode, epoch, loss_meter[mode], accuracy_meter[mode])

            # сохраняем результаты по всем эпохам
            loss_track[mode].append(loss_meter[mode].value())
            accuracy_track[mode].append(accuracy_meter[mode].value())
    return loss_track, accuracy_track


if __name__ == '__main__':
    # Добавьте эту строку для исправления проблем с multiprocessing
    freeze_support()
    # Считываем данные из файлов
    with open('dataset.csv', 'r', encoding='utf-8') as f:
        dataset = f.read().split('\n')

    for item in dataset[1:-1]:
        reviews.append(Review(item))

    for review in reviews:
        clear_text, text_words = preprocess(review.text)
        # print("!!!!!")
        # print(clear_text)
        # print(text_words)
        all_texts.append(clear_text)
        all_words.extend(text_words)
        labels.append(review.label)

    corpus = Counter(all_words)
    # Отсортируем слова по встречаемости
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)  # [:1000]
    # кодируем каждое слово - присваиваем ему порядковый номер
    vocab_to_int = {w: i + 1 for i, w in enumerate(corpus_)}

    # Кодируем все отзывы: последовательность слов --> последовательность чисел
    encoded_texts = []
    for sent in all_texts:
        encoded_texts.append([vocab_to_int[word] for word in sent.lower().split()
                              if word in vocab_to_int.keys()])

    encoded_labels = [1 if label == "positive" else 0 for label in labels]

    padded_texts = pad_text(encoded_texts, seq_length=200)

    train_set = TensorDataset(torch.from_numpy(padded_texts[:20000]),
                              torch.from_numpy(np.array(encoded_labels[:20000])))
    val_set = TensorDataset(torch.from_numpy(padded_texts[20000:]), torch.from_numpy(np.array(encoded_labels[20000:])))

    batch_size = 50

    # train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-5000, 5000])
    loaders = {'training': DataLoader(train_set, batch_size, pin_memory=True, num_workers=2, shuffle=True),
               'validation': DataLoader(val_set, batch_size, pin_memory=True, num_workers=2, shuffle=False)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_size = len(vocab_to_int) + 1
    embedding_dim = 100
    hidden_dim = 256
    num_layers = 1


    # Бинарная сериализация
    with open('model/vocab.bin', 'wb') as bin_file:
        pickle.dump(vocab_to_int, bin_file)

    # model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, num_layers)

    # Создайте новый экземпляр модели с теми же параметрами
    # model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, num_layers)
    # Загрузите веса из файла
    # model.load_state_dict(torch.load('model.pth'))
    # Убедитесь, что модель находится в режиме оценки
    # model.eval()

    # model.to(device)

    # optimizer = torch.optim.Adam(params=model.parameters())  # алгоритм оптимизации
    # lr = 0.001  # learning rate

    # loss_track, accuracy_track = trainval(model, loaders, optimizer, epochs=30)
    # torch.save(model.state_dict(), 'model.pth')


