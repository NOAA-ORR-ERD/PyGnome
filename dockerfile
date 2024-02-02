ARG PYTHON_VER=3.10

FROM registry.orr.noaa.gov/erd/centos-conda/miniforge-python$PYTHON_VER

# Args declared before the FROM need to be redeclared, don't remove this line
ARG PYTHON_VER

RUN yum install -y wget gcc make bzip2 gcc-c++ ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion tar

COPY ./ /pygnome/

RUN cd pygnome && conda install python=$PYTHON_VER \
                                --file conda_requirements.txt \
                                --file conda_requirements_build.txt \
                                --file oil_database/adios_db/conda_requirements.txt

# this was pinning things down too much for the webapi step
# RUN cd pygnome && conda install -y --file deploy_requirements.txt

RUN conda list

# adios_db requirements should already be there from the deploy_requirements file
# RUN cd pygnome/oil_database/adios_db && conda install --file conda_requirements.txt

RUN cd pygnome/py_gnome && python setup.py install
RUN cd pygnome/oil_database/adios_db && python -m pip install ./

# to check if they got installed properly
RUN python -c "import adios_db"
RUN python -c "import gnome"

