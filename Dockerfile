FROM tensorflow/tensorflow:2.14.0-gpu

WORKDIR /experiment

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /experiment

ENV PYTHONBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python" , "src/main.py"]