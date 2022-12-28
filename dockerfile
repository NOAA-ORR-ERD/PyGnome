ARG PYTHON_VER

FROM registry.orr.noaa.gov/erd/centos-conda/centos7-python$PYTHON_VER

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

# this kludge should no longer be required
# RUN conda update -y libgd

# only because this was giving us problems
# RUN python -c "import py_gd"

# adios_db requirements should already be there from the deploy_requirements file
# RUN cd pygnome/oil_database/adios_db && conda install --file conda_requirements.txt

RUN cd pygnome/py_gnome && python setup.py install
RUN cd pygnome/oil_database/adios_db && python -m pip install ./

# to check if they got installed properly
RUN python -c "import adios_db"
RUN python -c "import gnome"

