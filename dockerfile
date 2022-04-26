FROM registry.orr.noaa.gov/erd/centos-conda/centos7-python3.9

RUN yum update -y

RUN yum install -y wget gcc make bzip2 gcc-c++ ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion tar

COPY ./ /pygnome/

RUN cd pygnome && conda install --file conda_requirements.txt
RUN cd pygnome/py_gnome && python -m pip install ./

RUN cd pygnome/oil_database/adios_db && conda install --file conda_requirements.txt
RUN cd pygnome/oil_database/adios_db && python -m pip install ./
