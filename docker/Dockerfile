
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

COPY requirements.txt .
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y python3 python3-pip

RUN pip install --upgrade pip
RUN pip install -r requirements.txt && \
    pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


WORKDIR root
