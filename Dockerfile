# Dockerfile based on this tutorial: https://towardsdatascience.com/detectron2-the-basic-end-to-end-tutorial-5ac90e2f90e3

FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev python3-opencv

# Detectron2 prerequisites
RUN pip install torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

WORKDIR /car_detection
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY car_detection/ ./
COPY tests/ ./tests/
ENV PYTHONPATH "${PYTHONPATH}:/car_detection"

ENTRYPOINT ["python", "/car_detection/app/object_detection.py"]
