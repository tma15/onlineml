# onlineml
A C++ library of online machine learning algorithms.

## Install
```
git clone https://github.com/tma15/onlineml.git
cd onlineml
autoreconf -iv
./configure --prefix=/path/to/install
make
make install
```

## File format
```
<label> <fature>:<value> ... <feature>:<value>
```

- `label` .=. characters
- `feature` .=. characters
- `value` .=. real values

## Training a model
```
/path/to/install/bin/train_onlineml_model -e <NUM_EPOCH> -m <MODEL_FILE> <TESTING_FILE> <TRAINING_FILE>
```

Only perceptron algorithm is supported now.
