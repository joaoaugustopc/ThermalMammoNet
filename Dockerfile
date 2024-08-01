FROM tensorflow/tensorflow:latest-gpu

WORKDIR /experiment

COPY . .

ENV PYTHONBUFFERED=1

CMD ["python" , "main.py"]