from string import punctuation


def preprocess(text):
    """"
    Функция чтобы очистить текст отзыва от пунктуации и выделить все слова
    """
    clear_text = "".join([s.lower() for s in text if s not in punctuation])  # убираем пунктуацию
    text_words = clear_text.split()  # получаем массив слов
    return clear_text, text_words