ARG PYTHON_VER
FROM registry.orr.noaa.gov/erd/centos-conda/miniforge-python$PYTHON_VER

# Args declared before the FROM need to be redeclared, don't remove this line
ARG PYTHON_VER

RUN yum install -y wget gcc make bzip2 gcc-c++ ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion tar

COPY ./ /pygnome/
WORKDIR /pygnome/

RUN conda install python=$PYTHON_VER \
        --file conda_requirements.txt \
        --file conda_requirements_build.txt \
        --file oil_database/adios_db/conda_requirements.txt

RUN cd py_gnome && python setup.py install
RUN cd oil_database/adios_db && python -m pip install ./

# to check if they got installed properly
RUN python -c "import adios_db"
RUN python -c "import gnome"

