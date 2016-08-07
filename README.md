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

## Training a model
```
/path/to/install/bin/train_onlineml_model -e <NUM_EPOCH> -m <MODEL_FILE> <TESTING_FILE> <TRAINING_FILE>
```

Only perceptron algorithm is supported now.


## Use from C++
```
#include <stdio.h>

#include <vector>
#include <map>
#include <string>
#include <iostream>

#include <onlineml/learner/perceptron.h>

int main(int argc, char const* argv[]) {
    Learner* p = new Perceptron();

    std::vector< std::map<std::string, float> > x;
    std::vector< std::string > y;

    std::map<std::string, float> x1;
    x1.insert(std::map<std::string, float>::value_type("apple", 1.));
    x1.insert(std::map<std::string, float>::value_type("banana", 1.));
    x.push_back(x1);
    y.push_back("en");

    std::map<std::string, float> x2;
    x1.insert(std::map<std::string, float>::value_type("リンゴ", 1.));
    x1.insert(std::map<std::string, float>::value_type("バナナ", 1.));
    x.push_back(x2);
    y.push_back("ja");

    for (int t=0; t<3; t++) {
        p->fit(x, y);
    }

    int ret = p->predict(x2);
    std::cout << "label:" << ret << std::endl;
    const char* label = p->id2label(ret);
    printf("ret: %s (%d)\n", label, ret);
    return 0;
}

```

## Use from Python
To use `onlineml` from Python, you need to generate a Python
wrapper by SWIG.
```
cd onlineml
make python
```

```
import sys
sys.path.append("/path/to/onlineml/swig/")
import onlineml

def accuracy_score(x, y):
    n = float(len(x))
    m = 0.
    for i in range(n):
        if x[i]==y[i]:
            m += 1.
    return m/n


p = onlineml.Perceptron()

i = 0

### read training data and learn a model
for line in open(sys.argv[1]):
    sp = line.strip().split()
    i += 1
    y_i = sp[0]
    x_i = {}
    for elem in sp[1:]:
        sp_ = elem.split(":")
        ft = sp_[0]
        val = float(sp_[1])
        x_i[ft] = val

    if i % 1000 == 0:
        print(i)
    p.fit(onlineml.fvs([x_i]), onlineml.labels([y_i]))

p.save("test")

cls = onlineml.Classifier()
cls.load("test")

### read testing data and predict labels by learned model
y_pred = []
y_true = []
for line in open(sys.argv[2]):
    sp = line.strip().split()
    y_true_i = sp[0]
    x_i = {}
    for elem in sp[1:]:
        sp_ = elem.split(":")
        ft = sp_[0]
        val = float(sp_[1])
        x_i[ft] = val

    x_test = onlineml.fv(x_i)

    i = cls.predict(x_test)
    y_pred_i = cls.id2label(i)

    y_true.append(y_true_i)
    y_pred.append(y_pred_i)

    print(y_true_i, y_pred_i, accuracy_score(y_true, y_pred))

print accuracy_score(y_true, y_pred)
```
