import io

from flask import Flask, render_template, request, flash

from pdfcalib.model import PdfCAPredictor
from pdfcalib.text_utils import pdf2text

MAX_SEQ_LENGTH = 500
MAX_TEXT_LENGTH = 1000
DICT_CAT = {1: "позитивная",
            0: "нейтральная",
            2: "негативная"}

model = PdfCAPredictor("saved_model/lstm_model.pt",
                       "saved_model/vocab.obj",
                       DICT_CAT,
                       padding=MAX_SEQ_LENGTH
                       )

app = Flask(__name__)
app.secret_key = "something only you know"
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        sentiment = ""
    else:
        sentiment = None
        if 'image' not in request.files:
            flash('Нет файлов в запросе!')
        file = request.files['image']
        if file.filename == '':
            flash('Не выбран файл!')
        if allowed_file(file.filename):
            with io.BytesIO() as inmemoryfile:
                file.save(inmemoryfile)
                try:
                    text = pdf2text(inmemoryfile, MAX_TEXT_LENGTH)
                    if not text: raise ValueError("Empty text")
                except Exception:
                    flash("Файл не содержит текста или ошибка парсинга!")
                else:
                    sentiment = model.predict(text)
        else:
            flash('Неизвестное расширение файла!')
    return render_template('index.html', sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
