#!/bin/sh

sudo docker build -t test-image .

sudo docker run -v /home/makino/code:/mnt -it --name test-container test-image /bin/bash -c '
    git clone -b develop /mnt/onlineml && \
    cd onlineml && \
    autoreconf -iv && \
    ./configure && \
    make && \
    make install && \
    make python && \
    wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/mnist.bz2 && \
    wget https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2 && \
    bunzip2 mnist.bz2 mnist.t.bz2 && \
    time train_onlineml_model -e 2 -a ap -m model ./mnist ./mnist.t && \
    cd swig && \
    time python sample.py ../mnist ../mnist.t
'

sudo docker stop test-container
sudo docker rm test-container
