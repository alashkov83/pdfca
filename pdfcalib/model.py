import sys

sys.path.append(".")
import pickle

import torch
import torch.nn as nn

from .text_utils import add_padding, clean_text


def acc(pred: torch.tensor, label: torch.tensor) -> float:
    """
    Расчёт accurancy
    :param pred: предсказанные logits
    :param label: истинные метки классов
    :return: метрика
    """
    pred = torch.argmax(pred, axis=1)
    return torch.sum(pred == label).item()


class SentimentRNN(nn.Module):
    """
    Класс маодели
    """

    def __init__(self,
                 no_layers: int,
                 vocab_size: int,
                 hidden_dim: int,
                 embedding_dim: int,
                 output_dim: int,
                 drop_prob: float = 0.35):
        """

        :param no_layers: число LSTM слоёв
        :param vocab_size: размер словаря
        :param hidden_dim: размерность скрытого состояния
        :param embedding_dim: длина вектор-вложения
        :param output_dim: размерность выхода == число классов
        :param drop_prob: вероятность drop_out
        """
        super().__init__()

        self.hidden_dim = hidden_dim  # размер вектора скрытого состояния

        self.no_layers = no_layers  # число lstm слоёв
        self.vocab_size = vocab_size  # размер словаря+1 (для отбивки)

        # слой переода номеров токенов в их векторные представления
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        # выходной слой
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        """Прямой проход"""
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        prediction = self.fc(hidden)
        return prediction


class PdfCAPredictor:
    def __init__(self, checkpoint: str,
                 vocab: str,
                 dict_cat: dict,
                 padding: int = 200) -> None:
        """

        :param checkpoint: путь к файлу модели
        :param vocab: путь к файлу словаря
        :param dict_cat: словарь номер класса -> имя класса
        :param padding: величина отбивки == максимальное число токенов в тексте
        """
        self.dict_cat = dict_cat
        self.padding = padding
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(checkpoint)
        with open(vocab, 'rb') as f:
            self.vocab = pickle.load(f)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, text) -> str:
        """
        Метод получение предсказания
        :param text: входной текст
        :return: сентимент
        """
        text = clean_text(text)
        seq = [[self.vocab[word] for word in text.split()
                if word in self.vocab.keys()]]
        pad = torch.from_numpy(add_padding(seq, self.padding))
        inputs = pad.to(self.device)
        with torch.no_grad():
            output = self.model(inputs)
        return self.dict_cat[torch.argmax(output).item()]
