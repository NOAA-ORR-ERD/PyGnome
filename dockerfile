FROM oillibrary

RUN yum install -y wget gcc make bzip2 gcc-c++ ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion tar

COPY ./ /pygnome/

RUN cd pygnome && conda install --file conda_requirements_py3.txt
RUN cd pygnome/py_gnome && python setup.py develop

RUN cd pygnome/py_gnome/documentation && make html
