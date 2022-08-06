FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing \
    && apt-get install -y \
    build-essential \
    python-pip \
    libdc1394-22 \
    libssl1.0 \
    libgtk2.0-dev \
    libjpeg-turbo-progs \
    libimage-exiftool-perl \
    libexempi3 \
    libexiv2-dev \
    libz-dev \
    libexpat-dev \
    libjpeg-dev \
    libboost-filesystem-dev \
    libxrender-dev \
    language-pack-en \
    make \
    cython3 \
    python3-pip \
    python3-dev \
    vim \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# install python 3.7
RUN apt-get update \
	&& apt-get -y install python3.7 python3.7-dev \
	&& update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 \
	&& update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2 \
	&& update-alternatives  --set python /usr/bin/python3.7 \
	&& python -m pip install --upgrade setuptools pip wheel

#Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

#Dependencies
RUN pip install jupyter
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
        && rm -r /tmp/requirements.txt \
        && pip install --extra-index-url https://admin:%25YlCvk_3gcwLn5iid1j1@nexus.meero.dev/repository/meero-pypy/simple xmp_manager  \
        && pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html 


CMD /bin/bash


