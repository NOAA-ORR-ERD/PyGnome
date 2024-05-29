ARG PYTHON_VER
FROM registry.orr.noaa.gov/erd/centos-conda/miniforge-python$PYTHON_VER

# Args declared before the FROM need to be redeclared, don't delete this
ARG PYTHON_VER

RUN yum install -y libglib2.0-0 libxext6 libsm6 libxrender1 \
    wget gcc make bzip2 gcc-c++ chrpath patchelf \
    ca-certificates git mercurial subversion tar

COPY ./ /pygnome/
WORKDIR /pygnome/

RUN conda install python=$PYTHON_VER \
        --file py_gnome/conda_requirements.txt \
        --file py_gnome/conda_requirements_build.txt \
        --file oil_database/adios_db/conda_requirements.txt

RUN cd oil_database/adios_db && python -m pip install ./

RUN cd py_gnome && pip install ./

# to check if they got installed properly
#RUN python -c "import adios_db"
#RUN python -c "import gnome"
