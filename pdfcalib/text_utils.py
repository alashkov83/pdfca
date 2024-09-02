import re
from collections import Counter

import PyPDF2
import nltk
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords


def clean_text(text: str) -> str:
    """
    Очистка текстов от спецсимволов
    :param text: входная строка
    :return: выходная строка
    """
    text = text.lower()
    text = re.sub(r'[^\sа-яё0-9]', '', text)
    text = re.sub(r'\w*\d+\w*', '', text)
    text = re.sub('\s{2,}', " ", text)
    return text


def create_vocab(text_data, vocab_size: int) -> dict:
    """
    Создание словаря для токенизатора
    :param text_data: текстовые данные (список текстов или Series)
    :param vocab_size: желаемый размер словаря
    :return: словарь токенов
    """
    word_list = []

    stop_words = set(stopwords.words('russian'))  # стоп-слова
    for sent in text_data:
        sent = clean_text(sent)
        for word in sent.split():
            if word not in stop_words and word != '':
                word_list.append(word)

    counter_words = Counter(word_list)
    corpus = sorted(counter_words,
                    key=counter_words.get,
                    reverse=True)[:vocab_size]
    corpus_dict = {w: i + 1 for i, w in enumerate(corpus)}
    return corpus_dict


def tokenize(text_data, vocab: dict) -> list:
    """
    Тоненизация тексов на слова
    :param text_data: тексты
    :param vocab: словарь токенов
    :return: список токенизированных текстов
    """
    final_list = []
    for sent in text_data:
        sent = clean_text(sent)
        final_list.append([vocab[word] for word in sent.split()
                           if word in vocab.keys()])
    return final_list


def prepare_data(x_train, y_train, x_val, y_val, dict_size: int, dict_label: dict) -> tuple:
    """
    Подготовка данных для обучения
    :param x_train: тренировочный набор текстов
    :param y_train: тренировочный набор меток
    :param x_val: валидационный набор текстов
    :param y_val: валидционный набор меток
    :param dict_size: размер словаря
    :param dict_label: словарь меток
    """
    vocab = create_vocab(x_train, dict_size)

    final_list_train, final_list_test = tokenize(x_train, vocab), tokenize(x_val, vocab)

    encoded_train = [dict_label[label] for label in y_train]
    encoded_test = [dict_label[label] for label in y_val]
    return np.array(final_list_train, dtype='object'), np.array(encoded_train), np.array(final_list_test,
                                                                                         dtype='object'), np.array(
        encoded_test), vocab


def add_padding(sentences, seq_len: int) -> np.ndarray:
    """
    Огранмчения числа токенов в тексте и дабавление отбивки
    :param sentences: токенизированные тексты
    :param seq_len: максимальная длина токенизированного текста
    :return: токенизированные тексты
    """
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def pdf2text(file, max_text_size: int = 1000) -> str:
    """
    Парсинг файла PDF
    :param file: имя файла или файловый объект
    :param max_text_size: максимальная длина обрабатываемого текста
    :return: извлеченный текст
    """
    reader = PyPDF2.PdfReader(file)
    text = " ".join([reader.pages[i].extract_text() for i in range(len(reader.pages))])
    return text[:max_text_size]
