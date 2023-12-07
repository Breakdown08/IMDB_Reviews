import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader

from main import SentimentRNN, pad_text
from utils import preprocess

vocab_to_int = {}

def predict(model, review, seq_length=200):
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


if __name__ == "__main__":
    with open('model/vocab.bin', 'rb') as bin_file:
        vocab_to_int = pickle.load(bin_file)
    vocab_size = len(vocab_to_int) + 1
    embedding_dim = 100
    hidden_dim = 256
    num_layers = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, num_layers)

    # Создайте новый экземпляр модели с теми же параметрами
    model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, num_layers)
    # Загрузите веса из файла
    model.load_state_dict(torch.load('model/model.pth'))
    # Убедитесь, что модель находится в режиме оценки
    model.eval()
    model.to(device)
    review1 = "Twin Peaks is a very good film to watch with a family. Even five year old child will understand David Lynch masterpiece"
    review2 = "It made me cry"
    review3 = "It made me cry - I never seen such an awful acting before"
    review4 = "Vulgarity. Ringing vulgarity"
    review5 = "Garbage"
    review6 = "Its a beautiful movie"
    review7 = """I'm a big fan of Nolan's work so was really looking forward to this. I understood there would be some flipping in timelines and I'd need to concentrate. I didn't find this to be a problem at all and the storytelling was beautifully done. The acting was universally excellent. I saw a review saying Emily Blunt was rather OTT but I didn't find that at all.
I think my biggest gripe with the film may mean that I'm just getting old. I found the direction quite jarring with jump cuts galore. While it did keep things moving along apace, it was all rather exhausting. I also found the music and sound very very loud to the point of intrusion. Much like other Nolan films as it goes: Interstellar that I love, also had *very* loud music.
All in all this is a quality watch. It just left me longing for the days when so called 'cerebral' biopics, were a little more tranquil."""

    predict(model, review1)
    predict(model, review2)
    predict(model, review3)
    predict(model, review4)
    predict(model, review5)
    predict(model, review6)
    predict(model, review7)