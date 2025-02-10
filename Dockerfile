FROM tensorflow/tensorflow:2.14.0-gpu

WORKDIR /experiment

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir seaborn
RUN pip install --no-cache-dir ultralytics
RUN pip install --no-cache-dir opencv-python


COPY . /experiment

ENV PYTHONBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python" ,"-m", "main"]