# onlineml
A C++ implementation of online machine learning algorithms

## Install
```
cd onlineml
mkdir build
cd build
cmake ..
make
make install
```

## Training & evaluation
```
./onlineml-train --input_file <train_file> --max_epoch 10
./onlineml-evaluate --input_file <test_file> --model_dir .
```
