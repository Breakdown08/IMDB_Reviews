from multiprocessing import freeze_support

import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from review import Review
from matplotlib import pyplot as plt

import random
random.seed(33)
torch.manual_seed(0)
reviews = []
all_texts = []
all_words = []
labels = []


def preprocess(text):
    """"
    Функция чтобы очистить текст отзыва от пунктуации и выделить все слова
    """
    clear_text = "".join([s.lower() for s in text if s not in punctuation])  # убираем пунктуацию
    text_words = clear_text.split()  # получаем массив слов
    return clear_text, text_words


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


# print('Общее число отзывов: ', len(all_texts))
# print('Общее число слов: ', len(all_words))
# print('Первые 2 отзыва: ', all_texts[:2])
# print('Первые 5 слов: ', all_words[:5])


corpus = Counter(all_words)
# Отсортируем слова по встречаемости
corpus_ = sorted(corpus,key=corpus.get,reverse=True)#[:1000]
print('Самые частые слова: ', corpus_[:10])
# кодируем каждое слово - присваиваем ему порядковый номер
vocab_to_int = {w:i+1 for i,w in enumerate(corpus_)}
print('Уникальных слов: ', len(vocab_to_int))

# Кодируем все отзывы: последовательность слов --> последовательность чисел
encoded_texts = []
for sent in all_texts:
  encoded_texts.append([vocab_to_int[word] for word in sent.lower().split()
                                  if word in vocab_to_int.keys()])
print('Пример закодированного ревью: ', encoded_texts[0])

encoded_labels = [1 if label == "positive" else 0 for label in labels]

print('Число отзывов и число лейблов: ', len(all_texts), len(labels))


def pad_text(encoded_texts, seq_length):
    padded = []
    for text in encoded_texts:
        if len(text) >= seq_length:
            padded.append(text[:seq_length])
        else:
            padded.append([0] * (seq_length - len(text)) + text)

    return np.array(padded)


padded_texts = pad_text(encoded_texts, seq_length=200)
print('Пример padded review: ', padded_texts[0])


train_set = TensorDataset(torch.from_numpy(padded_texts[:20000]), torch.from_numpy(np.array(encoded_labels[:20000])))
val_set = TensorDataset(torch.from_numpy(padded_texts[20000:]), torch.from_numpy(np.array(encoded_labels[20000:])))

batch_size = 50

# train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-5000, 5000])
print('Размер обучающего и валидационного датасета: ', len(train_set), len(val_set))
loaders = {'training': DataLoader(train_set, batch_size, pin_memory=True,num_workers=2, shuffle=True),
           'validation':DataLoader(val_set, batch_size, pin_memory=True,num_workers=2, shuffle=False)}


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


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

vocab_size = len(vocab_to_int) + 1
embedding_dim = 100
hidden_dim = 256
num_layers = 1
model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, num_layers)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters())  # алгоритм оптимизации
lr = 0.001  # learning rate


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
    loss_track, accuracy_track = trainval(model, loaders, optimizer, epochs=30)

    plt.plot(accuracy_track['training'], label='train')
    plt.plot(accuracy_track['validation'], label='val')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()

    plt.plot(loss_track['training'], label='train')
    plt.plot(loss_track['validation'], label='val')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()


    def predict(model, review, seq_length=200):
        print(review)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _, words = preprocess(review.lower())
        encoded_words = [vocab_to_int[word] for word in words if word in vocab_to_int.keys()]
        padded_words = pad_text([encoded_words], seq_length)
        padded_words = torch.from_numpy(padded_words).to(device)
        bs = 1
        model.eval()
        zero_init = torch.zeros(num_layers, bs, hidden_dim).to(device)
        h = tuple([zero_init, zero_init])
        output = model(padded_words, h)
        pred = torch.round(output.squeeze())
        out = "This is a positive review." if pred == 1 else "This is a negative review."
        print(out, '\n')


    review1 = "Twin Peaks is a very good film to watch with a family. Even five year old child will understand David Lynch masterpiece"
    review2 = "It made me cry"
    review3 = "It made me cry - I never seen such an awful acting before"
    review4 = "Vulgarity. Ringing vulgarity"
    review5 = "Garbage"

    predict(model, review1)
    predict(model, review2)
    predict(model, review3)
    predict(model, review4)
    predict(model, review5)