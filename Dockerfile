FROM debian:latest

RUN apt-get update --fix-missing \
    && apt-get install -y eatmydata \ 
    && eatmydata apt-get install -y \
        wget \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        git \
        libfreetype6-dev \
        swig \
        mpich \
        pkg-config \
        gcc \
        wget \
        curl \
        vim \
    && echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
    && wget --quiet https://repo.continuum.io/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh \
    && /bin/bash ~/anaconda.sh -b -p /opt/conda \
    && rm ~/anaconda.sh
# Setup anaconda path
ENV PATH /opt/conda/bin:$PATH

# Enable UTF-8 locale
ENV LANG C.UTF-8

# RUN conda update -n base -c defaults conda

RUN conda update setuptools \
    && pip install\
        numpy>=1.19.1 \
        pandas>=1.1.1 \
        scikit-learn>=0.12 \
        matplotlib>=3.3.1 \
        seaborn>=0.11.0 \
        scipy>=1.7.1 \ 
        pingouin>=0.5.0 

RUN pip install jupyter \
    && echo 'jupyter() { if [[ $* == "notebook" ]]; then command jupyter notebook /mnt/ --port=9999 --no-browser --ip=0.0.0.0 --allow-root; else command jupyter "$@"; fi; }' >> /root/.bashrc

CMD ["/bin/bash", "-c", "source /root/.bashrc && [ -z $SRC_INSTALLED ] && { echo \"installing mb-cog-maps package\" && python -W ignore -m  pip install -e . && echo 'SRC_INSTALLED=1' >> /root/.bashrc; }; /bin/bash"]

# Set default working directory to repo mountpoint
WORKDIR /mnt

EXPOSE 9999