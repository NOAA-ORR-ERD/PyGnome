FROM oillibrary

RUN yum install -y wget gcc make bzip2 gcc-c++ ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion tar

RUN cd ./py_gnome

RUN conda install --file conda_requirements.txt
RUN python setup.py install

#RUN cd ./documentation && make html