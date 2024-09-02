from unittest import TestCase
from .text_utils import add_padding, clean_text, create_vocab
import numpy as np

class TestTextUtils(TestCase):
    def test_clean_text(self):
        text = ('На балконе спасатели обнаружили кота, который успел надышаться дымом и уже не подавал признаков жизни. '
                'Сотрудники провели успешную реанимацию животного и передали его хозяевам')
        ct = clean_text(text)
        self.assertEqual(ct, 'на балконе спасатели обнаружили кота который успел надышаться дымом и уже не подавал'
                             ' признаков жизни сотрудники провели успешную реанимацию животного и передали его хозяевам')

    def test_add_padding(self):
        tokens = [[1, 2, 4, 4, 5, 12]]
        true_pad = np.array([[0, 0, 0, 0, 1, 2, 4, 4, 5, 12]])
        pad = add_padding(tokens, 10)
        self.assertSequenceEqual(pad.tolist(), true_pad.tolist())

    def test_create_vocab(self):
        true_vocab = {'балконе': 1,
                      'дымом': 8,
                      'кота': 4,
                      'который': 5,
                      'надышаться': 7,
                      'обнаружили': 3,
                      'подавал': 9,
                      'признаков': 10,
                      'спасатели': 2,
                      'успел': 6}
        text = ['на балконе спасатели обнаружили кота который успел надышаться дымом и уже не подавал'
                ' признаков жизни сотрудники провели успешную реанимацию животного и передали его хозяевам']
        vocab = create_vocab(text, 10)
        self.assertEqual(vocab, true_vocab)

