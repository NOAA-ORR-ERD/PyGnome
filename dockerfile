ARG PYTHON_VER
FROM registry.orr.noaa.gov/erd/centos-conda/ubuntu/ubuntuforge-python$PYTHON_VER

# Args declared before the FROM need to be redeclared, don't delete this
ARG PYTHON_VER

RUN apt-get update
RUN apt-get upgrade -y

RUN apt-get install -y \
    g++ make libxext-dev libsm-dev libxrender-dev libglib2.0-0 \
    wget chrpath bzip2 tar \
    git patchelf ca-certificates

COPY ./ /pygnome/
WORKDIR /pygnome/

RUN conda install python=$PYTHON_VER \
    --file py_gnome/conda_requirements.txt \
    --file py_gnome/conda_requirements_build.txt \
#    --file oil_database/adios_db/conda_requirements.txt

# RUN cd oil_database/adios_db && python -m pip install ./

RUN cd py_gnome && python -m pip install ./

# to check if they got installed properly
#RUN python -c "import adios_db"
#RUN python -c "import gnome"
