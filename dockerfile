ARG PYTHON_VER

FROM registry.orr.noaa.gov/erd/centos-conda/centos7-python$PYTHON_VER

RUN yum update -y

RUN yum install -y wget gcc make bzip2 gcc-c++ ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion tar

COPY ./ /pygnome/

RUN cd pygnome && conda install python=$PYTHON_VER --file conda_requirements.txt
RUN conda list

RUN cd pygnome/py_gnome && python setup.py install
RUN python -c "import gnome"

RUN cd pygnome/oil_database/adios_db && conda install --file conda_requirements.txt
RUN cd pygnome/oil_database/adios_db && python -m pip install ./
