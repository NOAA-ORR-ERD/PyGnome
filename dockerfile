FROM registry.orr.noaa.gov/erd/centos-conda/centos7-python3.8

RUN yum update -y

RUN yum install -y wget gcc make bzip2 gcc-c++ ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion tar

COPY ./ /pygnome/

RUN cd pygnome && conda install --file conda_requirements.txt
RUN cd pygnome/py_gnome && python setup.py install
RUN cd pygnome/oil_database/adios_db && python setup.py install

RUN cd pygnome/py_gnome/documentation && make html
