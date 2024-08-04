FROM tensorflow/tensorflow:2.14.0-gpu

WORKDIR /experiment


RUN pip install --no-cache-dir scikit-learn

COPY . /experiment

ENV PYTHONBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python" ,"-m", "main.py"]