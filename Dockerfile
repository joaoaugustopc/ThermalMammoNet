FROM tensorflow/tensorflow:2.14.0-gpu

WORKDIR /experiment

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir seaborn
RUN pip install --no-cache-dir ultralytics
RUN pip install --no-cache-dir opencv-python
RUN pip install --no-cache-dir pytesseract Pillow
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir tqdm
RUN pip install "numpy<2"


# Configura a variável de ambiente TESSDATA_PREFIX (opcional, mas boa prática)
# O Tesseract procura os arquivos de idioma (treinamento) em TESSDATA_PREFIX/tessdata
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/

COPY . /experiment

ENV PYTHONBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python" ,"-m", "main"]