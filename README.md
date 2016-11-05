# onlineml
A C++ implementation of online machine learning algorithms

## Install
```
git clone https://github.com/tma15/onlineml.git
cd onlineml
autoreconf -iv
./configure --prefix=/path/to/install
make
make install
```

To install python wrapper,

```
make python
```

## File format
```
<label> <fature>:<value> ... <feature>:<value>
```

- `label` .=. characters
- `feature` .=. characters
- `value` .=. real value

## Training a model
A model can be trained with Iterative Parameter Mixture that is a parallel training method.
```
/path/to/install/bin/train_onlineml_model -a (a|ap) -e <NUM_EPOCH> -m <MODEL_FILE> <TRAINING_FILE> <TESTING_FILE>
```

- `-a`: learning algorithm
  - `p`: perceptron
  - `ap`: averaged perceptron
- `-e`: number of epoch
- `-m`: model file name
- `-p`: number of threads

## Testing a model
```
/path/to/install/bin/test_onlineml_model model <TESTING_FILE>
```
