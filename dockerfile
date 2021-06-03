FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3

RUN apt update && apt install -y apt-utils && \
    apt install -y \
    xli \
    vim \
    tmux \
    git \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    libgl1-mesa-glx

RUN pip3 install \
    opencv-python==4.5.1.48 \
    pandas \
    pillow \
    scipy